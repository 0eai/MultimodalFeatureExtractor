import re
from pathlib import Path

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_all_files_sorted(folder_path, file_extension="*"):
    folder = Path(folder_path)
    files = [path.as_posix() for path in list(folder.glob(file_extension))]
    sorted_files = sorted(files, key=lambda f: natural_sort_key(f.split('/')[-1]))
    return sorted_files
