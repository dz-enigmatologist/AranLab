import os
from openpyxl import Workbook

def write_headers_to_worksheet(worksheet):
    """Helper function to write headers to a worksheet"""
    headers = [
        'time(mins)', 'delta Rct-a', 'Rct-d', 'Cp1', 'Ph1', 
        'Slope 1', 'Slope 2', 'Slope 3', 'Slope 4', 'Slope 5', 
        'Angle', 'Cp_exp-a', 'Cp_exp-b', 'Ph_slope', 'Ph_peak', 
        'Area Cp', 'Area Ph', 'Area Slope', 'Area Rs-direct', 'Area Rs-Para',
        '','','linear_eq_m','linear_eq_b','Rs','delta Rct-i','Q','n'
    ]
    
    # Write headers to the first row
    for col_num, header in enumerate(headers, start=1):
        worksheet.cell(row=1, column=col_num, value=header)

def create_chips_folder_and_workbooks():
    # Folder name
    folder_name = "Processed Chips for PCAI1ia"
    abs_folder_path = os.path.abspath(folder_name)
    #print(abs_folder_path)
    
    # Workbook names
    workbook_names = [
        'Chip 21', 'Chip 22', 'Chip 23', 'Chip 26', 'Chip 27',
        'Chip 32', 'Chip 33', 'Chip 35', 'Chip 36', 'Chip 37',
        'Chip 39', 'Chip 40', 'Chip 41', 'Chip 43', 'Chip 44',
        'Chip 45', 'Chip 46', 'Chip 47', 'Chip 48', 'Chip 52',
        'Chip 53', 'Chip 54', 'Chip 55', 'Chip 56', 'Chip 57',
        'Chip 59'
    ]
    
    # Worksheet names
    worksheet_names = ["0pM_asso", "0pM_disso", "100pM_asso", "100pM_disso", "1nM_asso", "1nM_disso", "10nM_asso", "10nM_disso", "100nM_asso", "100nM_disso"]
    
    # Create folder (overwrite if exists)
    try:
        os.makedirs(folder_name, exist_ok=True)
        #print(f"Folder '{folder_name}' created.")
    except OSError as error:
        #print(f"Error creating folder: {error}")
        return
    
    # Create workbooks
    for name in workbook_names:
        try:
            # Create a new workbook
            wb = Workbook()
            
            # Remove the default sheet created by openpyxl
            default_sheet = wb.active
            if default_sheet is not None: # <--- ADD THIS CHECK
                wb.remove(default_sheet)
            else:
                print(f"Warning: No active sheet found in new workbook {name}. Skipping removal.")


            
            # Create all worksheets with specified names
            for sheet_name in worksheet_names:
                ws = wb.create_sheet(title=sheet_name)
                write_headers_to_worksheet(ws)
                #print(f"Write in : {sheet_name}")
            
            # Save with .xlsx extension
            file_path = os.path.join(folder_name, f"{name}.xlsx")
            wb.save(file_path)
            #print(f"Created workbook '{name}' with worksheets: {worksheet_names}")
            
        except Exception as e:
            print(f"Error creating workbook {name}: {e}")