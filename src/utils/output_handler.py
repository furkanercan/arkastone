import os
import json
import numpy as np
# from datetime import datetime

def create_output_folder(label="default"):
    """
    Create a timestamped output folder with an optional label.

    Args:
        base_folder (str): The base directory where the output folder will be created.
        label (str): A label to append to the folder name for better identification.

    Returns:
        str: The path of the created output folder.
    """
    base_folder = "output" 

    # Generate timestamp in the desired format
    # timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Create the folder name
    folder_name = label
    output_path = os.path.join(base_folder, folder_name)
    
    # Create the folder if it does not exist
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        return super().default(obj)

def save_config_to_folder(config, output_folder, config_name="savedconfig.json"):
    """
    Save a configuration dictionary to a JSON file in the specified folder.

    Args:
        config (dict): The configuration dictionary to save.
        output_folder (str): The folder where the configuration will be saved.
        config_name (str): The name of the configuration file.

    Returns:
        str: The path to the saved configuration file.
    """
    def compact_list(obj):
        """Convert lists to compact format for JSON serialization."""
        if isinstance(obj, list):
            return json.dumps(obj, separators=(", ", ":"))
        if isinstance(obj, dict):
            return {key: compact_list(value) for key, value in obj.items()}
        return obj

    compact_config = compact_list(config)

    config_path = os.path.join(output_folder, config_name)
    with open(config_path, "w") as f:
        f.write(
            json.dumps(compact_config, indent=4, cls=NumpyEncoder)
            .replace('"[', '[')  # Remove quotes around lists
            .replace(']"', ']')
        )
    # print(f"Configuration saved to {config_path}.")
    return config_path