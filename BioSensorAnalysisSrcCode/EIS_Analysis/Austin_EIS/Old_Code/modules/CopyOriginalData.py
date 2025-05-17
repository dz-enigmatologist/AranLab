def copy_dataframe_to_sheet(df, ws):
    """Writes the DataFrame into the worksheet ws starting at cell A1."""
    for col_idx, col in enumerate(df.columns, start=1):
        ws.cell(row=1, column=col_idx, value=col)
    for row_idx, row in enumerate(df.itertuples(index=False), start=2):
        for col_idx, value in enumerate(row, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

def copy_worksheet(source_ws, target_ws):
    """
    Copies cell values from source_ws (an openpyxl worksheet) to target_ws.
    (Note: This copies only cell values, not styles or formulas.)
    """
    for row in source_ws.iter_rows():
        for cell in row:
            target_ws[cell.coordinate].value = cell.value