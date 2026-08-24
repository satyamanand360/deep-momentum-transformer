import torch
import torch.nn as nn
import torch.nn.functional as F

class GLU(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        return self.fc1(x) * torch.sigmoid(self.fc2(x))

class GRN(nn.Module):
    """Gated Residual Network"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        return self.norm(res + x)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, d_model, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        # We assume input is (batch, seq, num_features). We embed each feature separately.
        # Since our inputs are continuous, we can just use a linear layer for each.
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        
        self.flattened_grn = GRN(num_features * d_model, dropout)
        self.softmax_linear = nn.Linear(num_features * d_model, num_features)
        
        self.single_feature_grns = nn.ModuleList([
            GRN(d_model, dropout) for _ in range(num_features)
        ])
        
    def forward(self, x):
        # x shape: (batch, seq, num_features)
        batch_size, seq_len, _ = x.shape
        
        # Embed each feature: output list of (batch, seq, d_model)
        embedded_features = []
        for i in range(self.num_features):
            feat = x[:, :, i].unsqueeze(-1) # (batch, seq, 1)
            embedded = self.feature_embeddings[i](feat)
            embedded_features.append(embedded)
            
        # Concatenate all embedded features
        # (batch, seq, num_features * d_model)
        concat_features = torch.cat(embedded_features, dim=-1)
        
        # Compute variable selection weights
        weight_network = self.flattened_grn(concat_features)
        weights = torch.softmax(self.softmax_linear(weight_network), dim=-1) # (batch, seq, num_features)
        
        # Process each feature through its own GRN and apply weights
        processed_features = []
        for i in range(self.num_features):
            processed = self.single_feature_grns[i](embedded_features[i]) # (batch, seq, d_model)
            weight = weights[:, :, i].unsqueeze(-1) # (batch, seq, 1)
            processed_features.append(processed * weight)
            
        # Sum over features
        # (batch, seq, d_model)
        output = torch.stack(processed_features, dim=-1).sum(dim=-1)
        return output, weights

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads
        
        self.q_layers = nn.Linear(d_model, d_model)
        self.k_layers = nn.Linear(d_model, d_model)
        # Value weights are shared across heads in TFT (Interpretable MHA)
        self.v_layer = nn.Linear(d_model, self.d_k)
        
        # W_H: output projection from d_k -> d_model (TFT paper requires this)
        self.W_H = nn.Linear(self.d_k, d_model)
        
        self.attention = None
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections for Q and K (per-head)
        q = self.q_layers(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_layers(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V is SHARED across heads: single projection (batch, seq, d_k)
        v_shared = self.v_layer(v) # (batch, seq, d_k)
        
        # Attention scores (per head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = torch.softmax(scores, dim=-1) # (batch, n_heads, seq, seq)
        self.attention = attn # Save for interpretability
        
        # Apply each head's attention to the SHARED V
        # v_shared: (batch, seq, d_k) -> (batch, 1, seq, d_k) for broadcast
        v_expanded = v_shared.unsqueeze(1) # (batch, 1, seq, d_k)
        context_per_head = torch.matmul(attn, v_expanded) # (batch, n_heads, seq, d_k)
        
        # Average across heads (Interpretable MHA: average instead of concatenate)
        context = context_per_head.mean(dim=1) # (batch, seq, d_k)
        
        # W_H output projection
        output = self.W_H(context) # (batch, seq, d_model)
        return output

class MomentumTransformer(nn.Module):
    def __init__(self, num_features=8, d_model=64, n_heads=4, dropout=0.4):
        super().__init__()
        
        self.vsn = VariableSelectionNetwork(num_features, d_model, dropout)
        
        # LSTM encoder for local processing
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            batch_first=True
        )
        
        self.lstm_norm = nn.LayerNorm(d_model)
        self.lstm_gate = GLU(d_model)
        
        # Interpretable Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_gate = GLU(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Linear(d_model, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_gate = GLU(d_model)
        
        # Final output layer predicting position size z_t
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x: (batch, seq, num_features)
        
        # 1. Variable Selection Network
        vsn_out, var_weights = self.vsn(x) # (batch, seq, d_model)
        
        # 2. LSTM
        lstm_out, _ = self.lstm(vsn_out) # (batch, seq, d_model)
        # Add & Norm with Gate
        lstm_out = self.lstm_norm(vsn_out + self.lstm_gate(lstm_out))
        
        # 3. Attention (Masked to prevent lookahead)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
        attn_out = self.attention(lstm_out, lstm_out, lstm_out, mask)
        # Add & Norm with Gate
        attn_out = self.attn_norm(lstm_out + self.attn_gate(attn_out))
        
        # 4. FFN
        ffn_out = self.ffn(attn_out)
        out = self.ffn_norm(attn_out + self.ffn_gate(ffn_out))
        
        positions = self.output_layer(out) # (batch, seq, 1)
        z = positions.squeeze(-1) # (batch, seq)
        
        return z, var_weights, self.attention.attention

if __name__ == "__main__":
    model = MomentumTransformer(num_features=8, d_model=64, n_heads=4)
    x = torch.randn(16, 252, 8) # batch=16, seq=252, features=8
    z, var_weights, attn = model(x)
    print(f"Position shape: {z.shape}") # Should be (16, 252)
    print(f"Var weights shape: {var_weights.shape}") # Should be (16, 252, 8)
    print(f"Attention shape: {attn.shape}") # Should be (16, 4, 252, 252)
