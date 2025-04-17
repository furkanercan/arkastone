import os
import re

def find_list_type_hints(root_dir: str):
    pattern = re.compile(r'List\[(int|float)\]')
    matches = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d != 'venv'] # Exclude virtual environment directories
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    for lineno, line in enumerate(file, start=1):
                        if 'List[' in line and pattern.search(line):
                            matches.append((filepath, lineno, line.strip()))
    return matches

# Example usage for the user's project directory
project_dir = "c:/Users/furka/Documents/communications_project"
list_type_hints = find_list_type_hints(project_dir)

import pandas as pd
df = pd.DataFrame(list_type_hints, columns=["File", "Line", "Code"])
print(df.to_string(index=False))