import subprocess
import shutil
from pathlib import Path
import sys

PRIVATE_REPO = Path(__file__).resolve().parent
PUBLIC_REPO = PRIVATE_REPO.parent / "arkastone-public"
SCRIPTS_DIR = PRIVATE_REPO / "scripts"

PUBLIC_PATHS = [
    ".github",
    "assets",
    "scripts/create_symlinks.py",
    "scripts/clean_symlinks.py",
    "configs",
    "examples",
    "src",
    "tests",
    "notebooks",
    "README.md",
    "LICENSE",
    "config.json5",
    ".gitignore",
    "main_dev.py",
    "requirements.txt"
]

def run_pytests():
    print("🧪 Running pytest...")
    result = subprocess.run(["pytest"], cwd=PRIVATE_REPO)
    return result.returncode == 0

def ignore_patterns(_, names):
    return {name for name in names if name == '__pycache__'}

def run_symlink_cleaner():
    print("🧹 Cleaning symlinks...")
    result = subprocess.run(["python", "clean_symlinks.py"], cwd=SCRIPTS_DIR)
    if result.returncode != 0:
        print("❌ clean_symlinks.py failed.")
        sys.exit(1)

def run_symlink_restorer():
    print("🔗 Restoring symlinks...")
    result = subprocess.run(["python", "create_symlinks.py"], cwd=SCRIPTS_DIR)
    if result.returncode != 0:
        print("❌ create_symlinks.py failed.")
        sys.exit(1)

def copy_public_files():
    print("📁 Copying public-safe files to arkastone...")

    for rel_path in PUBLIC_PATHS:
        src = PRIVATE_REPO / rel_path
        dst = PUBLIC_REPO / rel_path

        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=ignore_patterns,
                dirs_exist_ok=True
            )
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print("✅ Public files copied successfully.")
    
def main():
    run_symlink_cleaner()

    if run_pytests():
        copy_public_files()
    else:
        print("❌ Tests failed. Aborting copy.")
        run_symlink_restorer()
        sys.exit(1)

    run_symlink_restorer()

if __name__ == "__main__":
    main()