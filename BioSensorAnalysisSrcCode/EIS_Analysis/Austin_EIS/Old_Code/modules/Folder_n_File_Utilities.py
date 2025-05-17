import os
import re
from tkinter import Tk, filedialog

def select_folder():
    """Prompts the user to select a folder."""
    Tk().withdraw()
    return filedialog.askdirectory(title="Select Master Folder Containing Data")

def get_sheet_name(filename):
    """
    Derives a sheet name from the CSV filename by:
      - Removing a date prefix if the first 6 characters are numbers followed by an underscore.
      - Dropping the file extension.
      - Splitting on spaces and underscores.
      - Removing any token that matches "chip<number>" or equals "after".
      - Joining the remaining tokens with underscores.
    """
    # Remove date prefix if present
    filename = re.sub(r'^\d{6}_', '', filename)
    name = os.path.splitext(filename)[0]
    tokens = re.split(r'[\s_]+', name)
    filtered_tokens = [
        token for token in tokens 
        if not re.fullmatch(r'chip\d+', token, flags=re.IGNORECASE) and token.lower() != "after"
    ]
    return "_".join(filtered_tokens) if filtered_tokens else name

def create_analysis_folder(master_folder):
    """Creates an output folder in the master folder."""
    base_folder = os.path.join(master_folder, "EIS_DataAnalysis_Folder")
    unique_folder = base_folder
    count = 1
    while os.path.exists(unique_folder):
        unique_folder = f"{base_folder} ({count})"
        count += 1
    os.makedirs(unique_folder)
    return unique_folder

def match_chip_files(chip_data):
    """
    Matches chip files and separates valid and invalid chips.
    A valid chip has all 5 required file types.
    """
    valid_chips = {chip: files for chip, files in chip_data.items() if len(files) == 5}
    invalid_chips = {chip: files for chip, files in chip_data.items() if len(files) < 5}
    return valid_chips, invalid_chips

def log_and_print(message, log_file):
    print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

