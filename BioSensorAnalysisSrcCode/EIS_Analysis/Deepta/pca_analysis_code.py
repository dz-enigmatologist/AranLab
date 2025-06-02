import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tkinter import Tk, filedialog
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from itertools import combinations
from pandas.plotting import parallel_coordinates
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Suppress main tkinter window
Tk().withdraw()

# Open file dialog
file_paths = filedialog.askopenfilenames(
    title="Select Excel files for PCA analysis",
    filetypes=[("Excel files", "*.xlsx *.xls")]
)

# Create output folder with date
date_str = datetime.now().strftime("%Y-%m-%d")
output_dir = f"{date_str}_PCA_Analysis"
os.makedirs(output_dir, exist_ok=True)

# Loop through selected Excel files
for file_path in file_paths:
    try:
        excel_file = pd.ExcelFile(file_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        pdf_filename = f"{base_filename}_{date_str}_PCA_Analysis.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # If PDF exists from earlier run, delete it
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        with PdfPages(pdf_path) as pdf:
            for sheet_name in excel_file.sheet_names:
                try:
                    df = excel_file.parse(sheet_name)

                    # Skip empty sheets
                    if df.empty:
                        print(f"Skipping empty sheet: {sheet_name} in {base_filename}")
                        continue

                    # Drop fully empty columns
                    df_clean = df.dropna(axis=1, how='all')

                    if df_clean.shape[1] < 2:
                        print(f"Not enough data in sheet: {sheet_name} in {base_filename}")
                        continue

                    # Separate time and data
                    time_col = df_clean.iloc[:, 0]  # first column is time
                    data = df_clean.iloc[:, 1:]

                    # Keep only numeric columns
                    data = data.select_dtypes(include=[np.number])

                    if data.shape[1] < 2:
                        print(f"Not enough numeric columns in {sheet_name} of {base_filename}")
                        continue

                    # Normalize data (min-max scaling to [0, 1])
                    data_normalized = (data - data.min()) / (data.max() - data.min())
                    data_normalized = data_normalized.fillna(0)  # Handle divide-by-zero cases

                    # Save normalized data if needed later
                    normalized_array = data_normalized.to_numpy()

                    # Perform PCA
                    pca = PCA()
                    pca_result = pca.fit_transform(normalized_array)
                    explained_var = pca.explained_variance_ratio_
                    original_features = data.columns.tolist()
                    num_components = pca_result.shape[1]

                    # Explained variance plot
                    fig1, axs1 = plt.subplots(1, 1, figsize=(8, 5))
                    axs1.bar([f"PC{i+1}" for i in range(num_components)], explained_var)
                    axs1.set_title(f'Explained Variance - {sheet_name}')
                    axs1.set_xlabel('Principal Component')
                    axs1.set_ylabel('Explained Variance Ratio')
                    plt.tight_layout()
                    pdf.savefig(fig1)
                    plt.close(fig1)

                    # --- [SCATTER PLOTS DISABLED] ---
                    """
                    # Pairwise scatter plots
                    pairs = list(combinations(range(num_components), 2))
                    n_plots = len(pairs)
                    n_cols = 3
                    n_rows = int(np.ceil(n_plots / n_cols))
                    fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
                    axs2 = axs2.flatten()

                    for i, (x, y) in enumerate(pairs):
                        axs2[i].scatter(pca_result[:, x], pca_result[:, y], alpha=0.7)
                        fx = original_features[np.argmax(np.abs(pca.components_[x]))]
                        fy = original_features[np.argmax(np.abs(pca.components_[y]))]
                        axs2[i].set_title(f'PC{x+1} ({fx}) vs PC{y+1} ({fy})')
                        axs2[i].set_xlabel(f'PC{x+1}')
                        axs2[i].set_ylabel(f'PC{y+1}')

                    for j in range(i + 1, len(axs2)):
                        fig2.delaxes(axs2[j])

                    plt.tight_layout()
                    pdf.savefig(fig2)
                    plt.close(fig2)
                    """

                    # Parallel coordinates plot
                    pc_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(num_components)])
                    pc_df["Index"] = pc_df.index  # Add dummy category

                    fig3 = plt.figure(figsize=(12, 6))

                    # Use a color cycle to color each PC line
                    colors = plt.cm.get_cmap("tab10", num_components)
                    for i, col in enumerate(pc_df.columns[:-1]):  # Skip "Index"
                        plt.plot(pc_df["Index"], pc_df[col], label=col, color=colors(i), alpha=0.7)

                    plt.title(f'Parallel Coordinates (PCs) - {sheet_name}')
                    plt.xlabel("Sample Index")
                    plt.ylabel("PC Score")
                    plt.legend(title="Component", bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    pdf.savefig(fig3)
                    plt.close(fig3)

                except Exception as e:
                    print(f"Error processing sheet {sheet_name} in {base_filename}: {e}")

        print(f"Saved PCA analysis to {pdf_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
