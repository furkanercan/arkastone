# 🚀 Environment Setup Guide (Clean and Minimal)

This guide will help you start from scratch using **Python's built-in `venv`**, without Conda.

---

## 🧼 Step 0: Clean-Up (One-Time Only)

If you're starting fresh and want to remove old virtual environments, run:

```bash
# Deactivate any active environment
conda deactivate || deactivate

# Optional: Remove conda environments (except base)
conda env list
conda remove --name <env_name> --all

# Delete all `.venv` folders recursively (from home)
find ~ -type d -name ".venv" -exec rm -rf {} +

# Clear Conda and pip caches
conda clean --all --yes
pip cache purge

# Optional: Disable Conda auto-activation
conda config --set auto_activate_base false
```

Edit your `~/.bashrc` or `~/.zshrc` to remove or comment out Conda auto-init if not needed.

---

## 🐍 Step 1: Create and Activate Python Virtual Environment

From inside your project folder:

```bash
python3 -m venv .venv
```

Activate it:

```bash
# Linux/macOS:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

---

## 📦 Step 2: Install Required Packages

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt  # if you have it

# Or manually install:
pip install numpy pandas matplotlib  # example
```

---

## 📝 Step 3: Save Installed Packages

```bash
pip freeze > requirements.txt
```

This ensures the environment can be recreated easily.

---

## 💡 Tips

- Use `.venv/` in every project (don’t share one across projects).
- Add `.venv` to your `.gitignore`:
  ```
  .venv/
  ```

---

## 🔁 To Recreate Environment on Another Machine

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 🔁 Profile your code interactively

Run the code in your virtual environment with options:
-m cProfile -o profile.out
```bash
/home/furkan/Documents/communications_project/.venv/bin/python -m cProfile -o profile.out /home/furkan/Documents/communications_project/examples/tx_rx_chain_example_polar.py
```
Visualize:
```bash
snakeviz profile.out
```

## Generate distribution for local client

pyinstaller --onefile local_client.py

---

Happy coding! ⚡
