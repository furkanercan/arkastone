import os

# Define directories to skip
SKIP_DIRS = {'__pycache__'}

def create_symlink_in_tests(tests_dir, src_dir):
    # Walk through all subdirectories in the tests directory
    for root, dirs, files in os.walk(tests_dir):
        # Skip any directory listed in SKIP_DIRS
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        # Define the target path for the symbolic link
        symlink_target = os.path.join(root, 'src')
        
        # Only create the symlink if it doesn't already exist and not in a skipped folder
        if not os.path.exists(symlink_target):
            try:
                # Create the symbolic link pointing to the src directory
                os.symlink(src_dir, symlink_target)
                print(f"Created symlink: {symlink_target} -> {src_dir}")
            except Exception as e:
                print(f"Failed to create symlink in {root}: {e}")
        else:
            print(f"Symlink already exists in {root}.")

if __name__ == "__main__":
    # Define the path to your tests directory and the src directory
    tests_directory = 'tests'  
    src_directory = 'src'  

    # Ensure the source directory exists
    if os.path.exists(src_directory):
        create_symlink_in_tests(tests_directory, src_directory)
    else:
        print(f"Source directory {src_directory} does not exist!")
