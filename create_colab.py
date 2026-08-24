import nbformat as nbf
import glob

# Read all python files
files = ["data.py", "features.py", "dataset.py", "model.py", "train.py"]
code_contents = []
for f in files:
    with open(f, "r") as file:
        code_contents.append(file.read())

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Momentum Transformer: Colab Training\nThis notebook contains the complete pipeline (data downloading, feature engineering, model definition, and training) in a single file, making it easy to run on Google Colab with a GPU."),
    nbf.v4.new_markdown_cell("## 1. Install Dependencies\nRun this cell to install required packages."),
    nbf.v4.new_code_cell("!pip install yfinance pandas numpy torch tqdm scipy"),
    nbf.v4.new_markdown_cell("## 2. All-in-One Source Code\nWe consolidate the modular codebase into this single cell for Colab compatibility."),
]

combined_code = ""
for i, f in enumerate(files):
    combined_code += f"# {'='*40}\n# {f}\n# {'='*40}\n\n"
    # Remove local relative imports
    lines = code_contents[i].split('\n')
    filtered_lines = []
    skip = False
    for line in lines:
        if line.startswith('if __name__ == "__main__":'):
            skip = True
            continue
        if skip and (line.startswith(' ') or line == ''):
            continue
        elif skip:
            skip = False
            
        if "from dataset import get_dataloaders" in line or "from model import MomentumTransformer" in line or "from backtest import calculate_metrics" in line:
            continue
        filtered_lines.append(line)
    combined_code += '\n'.join(filtered_lines) + "\n\n"

# Remove the __main__ blocks from the intermediate modules so it doesn't execute prematurely
# Wait, a simpler way is to just define them, we can wrap the __main__ parts or replace them.
# Let's just put the combined code in a cell.
cells.append(nbf.v4.new_code_cell(combined_code))

cells.append(nbf.v4.new_markdown_cell("## 3. Execute Pipeline\nRun the following cell to download data, compute features, and start training."))
execution_code = """
import os

# 1. Download Data
tickers = [
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 
    'GLD', 'SLV', 'USO', 'UNG', 'DBA', 
    'VNQ', 'XLF', 'XLK', 'XLV', 'XLE',
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
    'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY', 'PEP',
    'KO', 'BAC', 'AVGO', 'TMO', 'COST', 'MCD', 'CSCO', 'WMT', 'CRM', 'DIS',
    'PFE', 'DHR', 'NKE', 'XOM', 'TXN', 'UNP', 'LIN', 'ADBE', 'VZ', 'PM',
    'NEE', 'RTX', 'QCOM', 'INTC', 'AMD', 'HON', 'UPS', 'AMGN', 'IBM', 'BA'
]
df_raw = download_data(tickers, "2005-01-01", "2024-01-01", "data/raw_data.csv")
print("Data downloaded!")

# 2. Compute Features
df_features = compute_features(df_raw)
os.makedirs("data", exist_ok=True)
df_features.to_csv("data/features.csv", index=False)
print("Features computed!")

# 3. Train Model
# The train_model() function defined above will automatically load data/features.csv, 
# instantiate the model, and train using PyTorch on GPU if available.
train_model()
"""
cells.append(nbf.v4.new_code_cell(execution_code))

nb['cells'] = cells
with open('colab_training.ipynb', 'w') as f:
    nbf.write(nb, f)
print("colab_training.ipynb created successfully.")
