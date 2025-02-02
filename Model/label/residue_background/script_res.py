import os
import shutil

# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Walk through every folder and file starting from the current directory
for root, dirs, files in os.walk(current_dir, topdown=False):
    for file in files:
        file_path = os.path.join(root, file)
        
        # If it's a .jpg file, delete it
        if file.lower().endswith('.jpg'):
            os.remove(file_path)
        
        # Otherwise, move the file to the current directory
        else:
            try:
                shutil.move(file_path, os.path.join(current_dir, file))
            except shutil.Error:
                # Skip if a file with the same name already exists in the target directory
                print(f"File {file} already exists in the destination, skipping.")
