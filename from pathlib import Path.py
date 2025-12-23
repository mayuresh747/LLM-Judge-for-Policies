from pathlib import Path

# Define the directory path. Use '.' for the current directory.
# Replace '.' with your specific directory path if needed, e.g., Path('/home/user/docs')
directory_path = Path('/Users/mayuri/Documents/Projects/Policy Conflict/New Documents/Topic 3 Public Transportation/DIR')

# Get a list of all files in the directory
# iterdir() iterates over all entries (files and directories)
filenames = [entry.name for entry in directory_path.iterdir() if entry.is_file()]

print(filenames)
