import re
import os

def find_valid_chips(master_folder):
    """
    Recursively searches the master folder for files.
    
    It gathers:
      - CSV files that include one of the keywords: "cyst", "dna", "water", "buffer"
      - Excel workbooks (".xlsx") that include the keyword "cas"
      
    All filenames must include a chip number (e.g. "Chip26").
    
    A valid chip has four CSV files (one per keyword) and one Excel workbook.
    Returns a dictionary mapping chip numbers (as strings) to a dictionary with keys:
      "cyst", "dna", "water", "buffer", and "cas".
    """
    chip_data = {}
    chip_pattern = re.compile(r'(?i)chip(\d+)')
    # Define required keywords for each file type.
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
                entry = chip_data.setdefault(chip_num, {})
                for key, keywords in required_patterns.items():
                    if key != "cas":
                        if f_lower.endswith('.csv') and all(kw in f_lower for kw in keywords):
                            entry[key] = full_path
                    else:
                        if f_lower.endswith('.xlsx') and all(kw in f_lower for kw in keywords):
                            entry[key] = full_path
    valid_chips = {chip: paths for chip, paths in chip_data.items() if all(k in paths for k in required_patterns)}
    return valid_chips