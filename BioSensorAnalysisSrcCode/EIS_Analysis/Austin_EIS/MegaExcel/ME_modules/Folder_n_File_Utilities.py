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

def find_valid_chips(master_folder):
    """
    Recursively searches the master folder for files.
    
    It gathers:
      - CSV files that include one of the keywords: "cyst", "dna", "water", "buffer"
      - Excel workbooks (".xlsx") that include the keyword "cas"
      
    All filenames must include a chip number (e.g. "Chip26").
    
    A valid chip must:
    1. Have four CSV files (one per keyword) and one Excel workbook
    2. Have a chip number in the approved list
    Returns a dictionary mapping chip numbers (as strings) to a dictionary with keys:
      "cyst", "dna", "water", "buffer", and "cas".
    """
    # Approved chip numbers
    valid_chip_numbers = {'21', '22', '23', '26', '27', '32', '33', '35', '36', '37', 
                         '39', '40', '41', '43', '44', '45', '46', '47', '48', '52', 
                         '53', '54', '55', '56', '57', '59'}
    
    chip_data = {}
    chip_pattern = re.compile(r'(?i)chip(\d+)')
    
    # Define required keywords for each file type
    required_patterns = {
        "water": ["water"],
        "buffer": ["buffer"],
        "dna": ["dna"],
        "cyst": ["cyst"],  # you can adjust as needed (e.g., "cystenine")
        "cas": ["cas"]
    }
    
    for root, dirs, files in os.walk(master_folder):
        for f in files:
            f_lower = f.lower()
            full_path = os.path.join(root, f)
            match = chip_pattern.search(f_lower)
            if match:
                chip_num = match.group(1)
                # Only process if chip number is in approved list
                if chip_num in valid_chip_numbers:
                    entry = chip_data.setdefault(chip_num, {})
                    for key, keywords in required_patterns.items():
                        if key != "cas":
                            if f_lower.endswith('.csv') and all(kw in f_lower for kw in keywords):
                                entry[key] = full_path
                        else:
                            if f_lower.endswith('.xlsx') and all(kw in f_lower for kw in keywords):
                                entry[key] = full_path
    
    # Final validation - must have all required files AND be in approved list
    valid_chips = {
        chip: paths 
        for chip, paths in chip_data.items() 
        if all(k in paths for k in required_patterns) and chip in valid_chip_numbers
    }
    
    return valid_chips