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

def find_valid_chips(master_folder, excel_only=False):
    """
    Recursively searches the master folder for files.

    Modes:
    - Full: Looks for CSVs and Excel files.
    - Excel-only: Only gathers Excel files with 'cas' and chip number.

    Returns:
        dict: Mapping chip numbers to file paths per data type.
    """
    valid_chip_numbers = {'21', '22', '23', '26', '27', '32', '33', '35', '36', '37',
                          '39', '40', '41', '43', '44', '45', '46', '47', '48', '52',
                          '53', '54', '55', '56', '57', '59'}

    chip_data = {}
    chip_pattern = re.compile(r'(?i)chip(\d+)')

    required_patterns = {
        "water": ["water"],
        "buffer": ["buffer"],
        "dna": ["dna"],
        "cyst": ["cyst"],
        "cas": ["cas"]
    }

    for root, dirs, files in os.walk(master_folder):
        for f in files:
            f_lower = f.lower()
            full_path = os.path.join(root, f)
            match = chip_pattern.search(f_lower)
            if match:
                chip_num = match.group(1)
                if chip_num in valid_chip_numbers:
                    entry = chip_data.setdefault(chip_num, {})
                    if f_lower.endswith('.xlsx') and all(kw in f_lower for kw in required_patterns["cas"]):
                        entry["cas"] = full_path
                    if not excel_only:
                        for key in ("water", "buffer", "dna", "cyst"):
                            if f_lower.endswith('.csv') and all(kw in f_lower for kw in required_patterns[key]):
                                entry[key] = full_path

    if excel_only:
        valid_chips = {chip: paths for chip, paths in chip_data.items() if "cas" in paths}
    else:
        valid_chips = {chip: paths for chip, paths in chip_data.items()
                       if all(k in paths for k in required_patterns)}

    return valid_chips
