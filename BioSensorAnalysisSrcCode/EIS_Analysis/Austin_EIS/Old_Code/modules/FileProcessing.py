import os
import pandas as pd
from openpyxl import load_workbook
import Folder_n_File_Utilities as fnf
import BioSensorAnalysisSrcCode.EIS_Analysis.Austin_EIS.modules.Analysis_CSV as ac
import plotting as pp
import CopyOriginalData as COD
import BioSensorAnalysisSrcCode.EIS_Analysis.Austin_EIS.modules.Analysis_xlsx as ax

def process_csv_file(file_path, writer, plot_data=False):
    """
    Processes a CSV file:
      - Reads the CSV.
      - Creates a new worksheet with a cleaned name.
      - Copies CSV content into the worksheet.
      - If the sheet name contains "cyst", "dna", "water", or "buffer", computes analysis on columns "Rs" and "X" and writes analysis in columns H onward.
      - Optionally plots the CSV data.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    filename = os.path.basename(file_path)
    sheet_name = fnf.get_sheet_name(filename)
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    keywords = ["cyst", "dna", "water", "buffer"]
    if any(kw in sheet_name.lower() for kw in keywords):
        analysis = ac.compute_analysis(df)
        if analysis:
            ws = writer.book[sheet_name]
            ac.add_analysis_to_sheet(ws, analysis, start_col=8)
    if plot_data:
        pp.plot_csv_data(df, sheet_name)

def process_excel_file(excel_file, writer, plot_data=False):
    """
    Processes an Excel workbook file:
      - Copies every worksheet from the source workbook into the new workbook.
      - For worksheets whose names do NOT contain any of the keywords 
        ["cyst", "dna", "water", "buffer", "association", "dissociation"],
        performs cycle analysis.
      - Otherwise, the sheet is copied exactly as-is.
    """
    try:
        source_wb = load_workbook(excel_file)
    except Exception as e:
        print(f"Error reading Excel workbook {excel_file}: {e}")
        return
    for sheet in source_wb.sheetnames:
        source_ws = source_wb[sheet]
        target_ws = writer.book.create_sheet(title=sheet)
        COD.copy_worksheet(source_ws, target_ws)
        # Define keywords that should prevent cycle analysis on the sheet.
        #skip_keywords = ["cyst", "dna", "water", "buffer", "association", "dissociation"]
        skip_keywords = ["cyst", "dna", "water", "buffer"]
        if not any(keyword in sheet.lower() for keyword in skip_keywords):
            ax.process_excel_cycles(target_ws)
        if plot_data:
            pp.plot_excel_sheet(source_ws)

def process_chip(chip_number, file_paths, output_folder, plot_data=False):
    """
    Processes one valid chip:
      - Creates a new workbook in the output folder named "Chip{chip_number}_EIS_Data Analysis.xlsx".
      - Creates a "summary" sheet.
      - Processes each CSV file (keys: "cyst", "dna", "water", "buffer") using process_csv_file.
      - Processes the Excel workbook file (key "cas") using process_excel_file.
    """
    workbook_name = os.path.join(output_folder, f"Chip{chip_number}_EIS_Data Analysis.xlsx")
    writer = pd.ExcelWriter(workbook_name, engine='openpyxl')
    pd.DataFrame().to_excel(writer, sheet_name="summary", index=False)
    for key in ["water", "buffer", "dna", "cyst"]:
        if key in file_paths:
            process_csv_file(file_paths[key], writer, plot_data=plot_data)
    if "cas" in file_paths:
        process_excel_file(file_paths["cas"], writer, plot_data=plot_data)
    writer.close()
    print(f"Processed Chip {chip_number} saved to {workbook_name}")