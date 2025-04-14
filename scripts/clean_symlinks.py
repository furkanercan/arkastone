import os

# Define directories to skip
SKIP_DIRS = {'__pycache__'}

def clean_symlinks_in_tests(tests_dir):
    # Walk through all subdirectories in the tests directory
    for root, dirs, files in os.walk(tests_dir):
        # Skip any directory listed in SKIP_DIRS
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        # Define the path for the symbolic link
        symlink_target = os.path.join(root, 'src')
        
        # Only remove the symlink if it exists and is a symlink
        if os.path.islink(symlink_target):
            try:
                # Remove the symbolic link
                os.remove(symlink_target)
                print(f"Removed symlink: {symlink_target}")
            except Exception as e:
                print(f"Failed to remove symlink in {root}: {e}")
        else:
            print(f"No symlink found in {root}.")

if __name__ == "__main__":
    # Define the path to your tests directory
    tests_directory = 'tests'  # Adjust this if the 'tests' directory is elsewhere

    clean_symlinks_in_tests(tests_directory)
