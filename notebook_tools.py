"""
notebook_tools.py — helpers for the create_*.py scripts that build the
single-file Colab / Kaggle notebooks from the repo's modules.
"""
LOCAL_MODULES = {"data", "features", "portfolio", "dataset", "model",
                 "train", "backtest", "live_data", "live_test"}


def _is_local_import(line):
    s = line.strip()
    return any(s.startswith(f"from {m} import") or s == f"import {m}" for m in LOCAL_MODULES)


def strip_module(code):
    """Drop `if __name__ == "__main__":` blocks and imports of the repo's own
    modules (including parenthesised multi-line ones), since everything is
    pasted into one notebook cell."""
    out, skip_main, skip_import = [], False, False
    for line in code.split("\n"):
        if line.startswith('if __name__ == "__main__":'):
            skip_main = True
            continue
        if skip_main:
            if line.startswith((" ", "\t")) or line == "":
                continue
            skip_main = False
        if skip_import:
            if ")" in line:
                skip_import = False
            continue
        if _is_local_import(line):
            if "(" in line and ")" not in line:
                skip_import = True
            continue
        out.append(line)
    return "\n".join(out)


def inline_modules(files):
    """Concatenate the given repo files into one block of source code."""
    combined = ""
    for fname in files:
        with open(fname) as fh:
            code = fh.read()
        combined += f"# {'=' * 50}\n# {fname}\n# {'=' * 50}\n\n" + strip_module(code).strip() + "\n\n"
    return combined
