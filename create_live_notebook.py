import nbformat as nbf

nb = nbf.v4.new_notebook()

# Read source files needed for live inference
files = ["features.py", "dataset.py", "model.py"]
code_contents = {}
for f in files:
    with open(f, "r") as fh:
        code_contents[f] = fh.read()

# Helper to strip __main__ blocks and cross-module imports
def clean(code):
    lines = code.split('\n')
    filtered, skip = [], False
    for line in lines:
        if line.startswith('if __name__ == "__main__":'):
            skip = True; continue
        if skip and (line.startswith(' ') or line == ''):
            continue
        elif skip:
            skip = False
        if any(x in line for x in [
            "from dataset import", "from model import",
            "from backtest import", "from features import"
        ]):
            continue
        filtered.append(line)
    return '\n'.join(filtered)

cells = []

# ── Title ──────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "# Momentum Transformer — Live Inference (2024-2026)\n"
    "Downloads fresh data up to **today**, runs the trained model, and shows:\n"
    "- Today's recommended portfolio positions ranked by signal strength\n"
    "- Live 90-day performance metrics vs benchmark\n\n"
    "> **Prerequisite:** Upload your trained `best_model.pth` to Kaggle as a Dataset input."
))

# ── Install ─────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1. Install Dependencies"))
cells.append(nbf.v4.new_code_cell("!pip install -q yfinance pandas numpy torch tqdm scipy"))

# ── Copy model from Kaggle input ────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
    "## 2. Load Trained Model\n"
    "Upload `best_model.pth` as a Kaggle Dataset. It will appear under `/kaggle/input/`."
))
cells.append(nbf.v4.new_code_cell(
    "import os, shutil\n"
    "\n"
    "# Kaggle datasets appear under /kaggle/input/<dataset-name>/\n"
    "# Adjust the path below to match your dataset name\n"
    "model_src = '/kaggle/input/momentum-transformer-model/best_model.pth'\n"
    "os.makedirs('models', exist_ok=True)\n"
    "if os.path.exists(model_src):\n"
    "    shutil.copy(model_src, 'models/best_model.pth')\n"
    "    print('Model copied to models/best_model.pth')\n"
    "else:\n"
    "    # Try to find it automatically\n"
    "    import glob\n"
    "    matches = glob.glob('/kaggle/input/**/*.pth', recursive=True)\n"
    "    if matches:\n"
    "        shutil.copy(matches[0], 'models/best_model.pth')\n"
    "        print(f'Model found and copied from: {matches[0]}')\n"
    "    else:\n"
    "        print('WARNING: best_model.pth not found. Check your Kaggle dataset input.')\n"
))

# ── Combined source code ────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3. Source Code (Architecture + Dataset)"))
combined = ""
for fname in ["features.py", "dataset.py", "model.py"]:
    combined += f"# {'='*50}\n# {fname}\n# {'='*50}\n\n"
    combined += clean(code_contents[fname]) + "\n\n"
cells.append(nbf.v4.new_code_cell(combined))

# ── Download Live Data ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 4. Download Live Data (2023 → Today)"))
cells.append(nbf.v4.new_code_cell(
    "import yfinance as yf\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "from datetime import datetime\n"
    "\n"
    "TICKERS = [\n"
    "    'SPY','QQQ','IWM','EFA','EEM','TLT','IEF','SHY','LQD','HYG',\n"
    "    'GLD','SLV','USO','UNG','DBA','VNQ','XLF','XLK','XLV','XLE',\n"
    "    'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',\n"
    "    'JPM','V','PG','MA','HD','CVX','MRK','ABBV','LLY','PEP',\n"
    "    'KO','BAC','AVGO','TMO','COST','MCD','CSCO','WMT','CRM','DIS',\n"
    "    'PFE','DHR','NKE','XOM','TXN','UNP','LIN','ADBE','VZ','PM',\n"
    "    'NEE','RTX','QCOM','INTC','AMD','HON','UPS','AMGN','IBM','BA'\n"
    "]\n"
    "\n"
    "today = datetime.today().strftime('%Y-%m-%d')\n"
    "print(f'Downloading data: 2023-01-01 to {today}')\n"
    "raw = yf.download(TICKERS, start='2023-01-01', end=today)\n"
    "raw = raw.stack(level=1).rename_axis(['Date','Ticker']).reset_index()\n"
    "\n"
    "os.makedirs('data', exist_ok=True)\n"
    "raw.to_csv('data/live_raw.csv', index=False)\n"
    "print(f'Downloaded {len(raw)} rows')\n"
    "\n"
    "live_features = compute_features(raw)\n"
    "live_features.to_csv('data/live_features.csv', index=False)\n"
    "print(f'Features computed: {live_features.shape}')\n"
    "live_features.head()\n"
))

# ── Run Live Inference ──────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 5. Run Live Inference"))
cells.append(nbf.v4.new_code_cell(
    "import torch\n"
    "from torch.utils.data import DataLoader\n"
    "\n"
    "SEQ_LENGTH    = 252\n"
    "VOL_TARGET    = 0.30\n"
    "COST_BPS      = 3\n"
    "device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print(f'Using device: {device}')\n"
    "\n"
    "# Load model\n"
    "model = MomentumTransformer(num_features=8, d_model=64, n_heads=4)\n"
    "state_dict = torch.load('models/best_model.pth', map_location=device)\n"
    "missing, _ = model.load_state_dict(state_dict, strict=False)\n"
    "if missing: print(f'[INFO] New layers randomly initialized: {missing}')\n"
    "model.to(device).eval()\n"
    "print('Model loaded!')\n"
    "\n"
    "# Build dataset — drop rows where Target is NaN (last row = today, no known return yet)\n"
    "df_hist = live_features.dropna(subset=['Target_Return']).copy()\n"
    "dataset = MomentumDataset(df_hist, seq_length=SEQ_LENGTH, step=1)\n"
    "loader  = DataLoader(dataset, batch_size=256, shuffle=False)\n"
    "print(f'Running inference on {len(dataset):,} sequences...')\n"
    "\n"
    "tgt_vol_daily = VOL_TARGET / np.sqrt(252)\n"
    "results = []\n"
    "\n"
    "with torch.no_grad():\n"
    "    for batch in loader:\n"
    "        X       = batch['X'].to(device)\n"
    "        y       = batch['y'].to(device)\n"
    "        vol     = batch['vol'].to(device)\n"
    "        z_seq, _, _ = model(X)\n"
    "        z   = z_seq[:, -1]\n"
    "        y_t = y[:, -1]\n"
    "        v_t = vol[:, -1]\n"
    "        scaled_z = z * (tgt_vol_daily / (v_t + 1e-8))\n"
    "        for i in range(len(z)):\n"
    "            results.append({\n"
    "                'Date':       batch['date'][i],\n"
    "                'Ticker':     batch['ticker'][i],\n"
    "                'z':          z[i].item(),\n"
    "                'scaled_z':   scaled_z[i].item(),\n"
    "                'raw_return': y_t[i].item(),\n"
    "                'vol':        v_t[i].item()\n"
    "            })\n"
    "\n"
    "res = pd.DataFrame(results)\n"
    "res['Date'] = pd.to_datetime(res['Date'])\n"
    "res = res.sort_values(['Ticker','Date'])\n"
    "\n"
    "# Transaction costs\n"
    "res['prev_z']   = res.groupby('Ticker')['scaled_z'].shift(1).fillna(0)\n"
    "res['turnover'] = (res['scaled_z'] - res['prev_z']).abs()\n"
    "cost_dec        = COST_BPS / 10000.0\n"
    "res['gross']    = res['scaled_z'] * res['raw_return']\n"
    "res['net']      = res['gross'] - res['turnover'] * cost_dec\n"
    "\n"
    "print('Inference complete!')\n"
    "print(f'Covered {res.Date.min().date()} to {res.Date.max().date()}')\n"
))

# ── Today's Positions ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 6. Today's Portfolio Positions"))
cells.append(nbf.v4.new_code_cell(
    "today_pos = res.groupby('Ticker').last().reset_index()[['Ticker','z','scaled_z']]\n"
    "today_pos = today_pos.sort_values('z', ascending=False)\n"
    "\n"
    "def direction(z):\n"
    "    if z > 0.15:  return 'LONG  [^]'\n"
    "    elif z < -0.15: return 'SHORT [v]'\n"
    "    return 'FLAT  [-]'\n"
    "\n"
    "today_pos['Direction'] = today_pos['z'].apply(direction)\n"
    "today_pos.to_csv('results/live_positions.csv', index=False)\n"
    "\n"
    "print(f\"TODAY'S POSITIONS (as of {res.Date.max().date()})\")\n"
    "print('-' * 50)\n"
    "print(f\"{'Ticker':<10} {'Signal z':>10} {'Direction':>12}\")\n"
    "print('-' * 35)\n"
    "for _, row in today_pos.iterrows():\n"
    "    print(f\"{row['Ticker']:<10} {row['z']:>10.4f} {row['Direction']:>12}\")\n"
))

# ── 90-Day Live Performance ─────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 7. Live 90-Day Performance vs Benchmark"))
cells.append(nbf.v4.new_code_cell(
    "def metrics(returns):\n"
    "    ann = 252\n"
    "    r = returns.mean()*ann; v = returns.std()*np.sqrt(ann)\n"
    "    sh = r/(v+1e-8)\n"
    "    down = returns[returns<0].std()*np.sqrt(ann)\n"
    "    so = r/(down+1e-8)\n"
    "    cum = (1+returns).cumprod()\n"
    "    dd = ((cum-cum.cummax())/cum.cummax()).min()\n"
    "    return {'Ann Return': r, 'Ann Vol': v, 'Sharpe': sh,\n"
    "            'Sortino': so, 'Max DD': dd, 'Win Rate': (returns>0).mean()}\n"
    "\n"
    "latest = res['Date'].max()\n"
    "cutoff = latest - pd.Timedelta(days=90)\n"
    "rec    = res[res['Date'] > cutoff]\n"
    "\n"
    "port  = rec.groupby('Date')['net'].mean()\n"
    "bench = rec.groupby('Date')['raw_return'].mean()\n"
    "\n"
    "# Apply vol targeting\n"
    "lev  = VOL_TARGET / (port.std()*np.sqrt(252) + 1e-8)\n"
    "port = port * lev\n"
    "\n"
    "sm = metrics(port)\n"
    "bm = metrics(bench)\n"
    "\n"
    "print(f'LIVE 90-DAY: {cutoff.date()} to {latest.date()}')\n"
    "print(f'{\"Metric\":<20} {\"Strategy\":>12} {\"Benchmark\":>12}')\n"
    "print('-'*46)\n"
    "for k in sm:\n"
    "    sv, bv = sm[k], bm[k]\n"
    "    pct = k in ['Ann Return','Ann Vol','Max DD','Win Rate']\n"
    "    if pct:\n"
    "        print(f'{k:<20} {sv*100:>11.2f}%  {bv*100:>11.2f}%')\n"
    "    else:\n"
    "        print(f'{k:<20} {sv:>12.4f}  {bv:>12.4f}')\n"
    "\n"
    "os.makedirs('results', exist_ok=True)\n"
    "port.to_csv('results/live_portfolio_returns.csv')\n"
    "bench.to_csv('results/live_benchmark_returns.csv')\n"
    "print('\\nSaved: results/live_positions.csv')\n"
    "print('Saved: results/live_portfolio_returns.csv')\n"
))

# ── Equity Curve Plot ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 8. Live Equity Curve"))
cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 6))\n"
    "strat_eq  = (1 + port).cumprod()\n"
    "bench_eq  = (1 + bench).cumprod()\n"
    "\n"
    "ax.plot(strat_eq.index, strat_eq.values, label='Momentum Transformer (30% Vol)', color='#00C49A', linewidth=2)\n"
    "ax.plot(bench_eq.index, bench_eq.values, label='Benchmark (Equal Weight)', color='#FF6B6B', linewidth=2, linestyle='--')\n"
    "ax.set_title('Live Performance: Momentum Transformer vs Benchmark', fontsize=14, fontweight='bold')\n"
    "ax.set_xlabel('Date')\n"
    "ax.set_ylabel('Cumulative Return')\n"
    "ax.legend()\n"
    "ax.grid(alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/live_equity_curve.png', dpi=150)\n"
    "plt.show()\n"
    "print('Chart saved: results/live_equity_curve.png')\n"
))

nb['cells'] = cells
with open('live_inference.ipynb', 'w') as f:
    nbf.write(nb, f)
print("live_inference.ipynb created successfully!")
