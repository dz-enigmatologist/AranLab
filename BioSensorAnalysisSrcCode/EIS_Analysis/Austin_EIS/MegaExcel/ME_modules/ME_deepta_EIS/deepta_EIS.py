import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltTxt
import matplotlib.pyplot as pltFit
import matplotlib.pyplot as pltTbl
import numpy as np
from tkinter import filedialog, Tk, Listbox, MULTIPLE, Button, END, simpledialog, messagebox, Entry, Label, Button
from scipy.optimize import curve_fit
from circle_fit import taubinSVD,plot_data_circle
from matplotlib.backends.backend_pdf import PdfPages
import math
from matplotlib.patches import Circle


# Function to manually load file
def manualloadfile():
    folderpath = "Users\\AcquisitionStation\\OneDrive - DB, Inc\\Documents\\Electrochemistry\\Chip Test"
    #folderpath = "/Users/deeptabharadwaj/Documents/lab/eis_analysis_code/data_to_process_aug19"
    folderpath = "/Users/deeptabharadwaj/Documents/lab/aranLab/eis_analysis_code/oct3Onwards_data/dataCheck_oct14/cas_association_nov20"
    root = Tk()
    #root.withdraw()  # Hide the root window
    root.filename = filedialog.askopenfilename(
        initialdir=folderpath, 
        title="Select file", 
        filetypes=(("Excel files", "*.xlsx"),("all files", "*.*"))
    )
    root.destroy()
    return root.filename, pd.ExcelFile(root.filename, engine='openpyxl')

# Function to select sheets
def select_sheets(xls):
    root = Tk()
    root.title('Select Sheets')

    listbox = Listbox(root, selectmode=MULTIPLE)
    listbox.pack(padx=10, pady=10)

    for sheet in xls.sheet_names:
        listbox.insert(END, sheet)

    def on_select():
        global selected_sheets
        selected_sheets = [listbox.get(i) for i in listbox.curselection()]
        root.destroy()

    button = Button(root, text='Select', command=on_select)
    button.pack(pady=10)

    root.mainloop()

def find_derivative_zero_crossing(x_values):
    """Find the index where the derivative of the x_values crosses zero."""
    dx = np.diff(x_values)
    zero_crossings = np.where(np.diff(np.sign(dx)))[0]
    derivative_zero = df['X'][zero_crossings+1]
    min_value = np.min(derivative_zero)
    min_index = np.argmin(derivative_zero)
    resp_idx = zero_crossings[min_index]+ 1
    if len(zero_crossings) == 0:
        return None

    return min_value, resp_idx

# Define the Randles circuit model
def randles_circuit(f, Rs, Rct, Q, n):
    omega = 2 * np.pi * f
    Zcpe = Q * (1j * omega) ** n
    Z = Rs + 1 / (1 / Rct + 1 / Zcpe)
    return np.real(Z), np.imag(Z)

# Define a function to fit the model to the data
def fit_eis_data(frequencies, Z_real, Z_imag):
    initial_guess = [10, 100, 1e-6, 0.9]
    Z_measured = np.concatenate((Z_real, Z_imag))
    
    popt, pcov = curve_fit(
        lambda f, Rs, Rct, Q, n: np.concatenate(randles_circuit(f, Rs, Rct, Q, n)),
        frequencies, Z_measured, p0=initial_guess)
    
    return popt, pcov

def is_already_opened(filename):
     if os.path.exists(filename):
         try:
             f = open(filename, 'a')
             f.close()
         except IOError:
             return True
     return False
def main():
    # Load the data
    file_path, xls = manualloadfile()

    # Extract initial file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Initialize file to write into
    if is_already_opened(f"results_{file_name}.pdf"):
        print(f"Please close file results_{file_name}.pdf and rerun")
        sys.exit(0)
    pdfFile = PdfPages(f'results_{file_name}.pdf')

    # Select sheets
    selected_sheets = []
    select_sheets(xls)


    # Define the default axis limits
    default_limits = {'xlim': (0, 1500), 'ylim': (0, 5000)}

    # Define the margin
    margin = 00

    # To store the last sections for the combined plot and plots
    last_sections = []
    figIncub = []
    plotName = []
    slopeList_RCT = []
    interceptList_RCT = []
    totalTime = 30 #total time of experiment
    results = pd.DataFrame(columns=['SheetName', 'Start Index', 'End Index','Set Number','Time','Lowest Value','diameter','Max X','Cp'])
    plots = {}


    #Test
    first_run = 0

    #To store the diameters of each section

    # Loop through each selected sheet in the Excel file
    for sheet_name in selected_sheets:
        set_number = 0
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name)
        
        # Define sections based on frequency ranges (e.g., 100-200000 in increments)
        maxFrequency = 200000
        maxFreqIndices = df.loc[df['Frequency(Hz)'] == maxFrequency].index
        maxFreqIndices = maxFreqIndices.to_numpy()
        maxFreqIndices = np.append(-1, maxFreqIndices)

        time_slots = np.linspace(1.5, totalTime, len(maxFreqIndices) - 1)
        diameters = []

        txt = "Processing Sheet: "+sheet_name+"It has the \n1. Raw Data graphs \n2. Circular Portions of graph \n3. Fitted Circle \n4. RCT vs Time \n5. Max X vs Time \n6. CP vs Time"
        pltTxt.axis('off')
        pltTxt.text(x=0, y=0.8, s=txt)
        pdfFile.savefig()

        
        # Initialize plot
        plt.figure(figsize=(15,10))

        # Plot each section
        plots[sheet_name] = [[],[],[]]
        for i, start in enumerate(maxFreqIndices):
            first_run = 1
            set_number = set_number + 1
            
            if i == len(maxFreqIndices) - 1:
                break
            section_df = df.iloc[maxFreqIndices[i] + 1:maxFreqIndices[i + 1] + 1]
            section_df_start_index = maxFreqIndices[i]
            section_df_end_index = maxFreqIndices[i + 1]

            frequencies = section_df['Frequency(Hz)'].values
        
            ##print(section_df)

            ## To find the minimun point in graph so we can split the graph into circle and line
            ## Keep checking the X values. As soon as it starts increasing catch that point as the beginning of semi circle. 
            ## Not a great way to find it, but hopefully the point it starts increasing is the point of start of semi circle
            #print(section_df)

            row = -section_df.iloc[1]['X']
            found_min = 0
            notAsemicircle = 0
            for index in range(0,len(section_df['X']-1)):
                if index == len(section_df['X']) - 1:
                    # If we havent found the increasing point and we have reached end of searching point of increase
                    notAsemicircle = 1
                    break;
                if -section_df.iloc[index]['X'] < -section_df.iloc[index+1]['X']:
                    row = -section_df.iloc[index]['X']
                    break
            

            # Split at index with mininum value 
            line_section_df = section_df.iloc[:index]  # Rows before index 
            circle_section_df = section_df.iloc[index:]  # Rows from index
    
            # convert circle_section,f into coo,nates,r circle-fit
            circle_coordinates = circle_section_df[['Rs','X']].to_numpy()
            
            # Fit a circle https://pypi.org/project/circle-fit/
            r = 0
            if (notAsemicircle == 0):
                xc, yc, r, sigma = taubinSVD(circle_coordinates)
            # Get the diameter and add to the list
            diameters.append(2*r)
            #print("Circle fit done")
            #print("Radius of circle")
            #print(r)


            # Plot the fitted circle
            positions = []
            if (notAsemicircle == 0):
                #plot_data_circle(circle_coordinates, xc, yc, r)


                t=0
                #positions = []
                while t <= 2 * math.pi:
                    positions.append((r * math.cos(t) + xc, r * math.sin(t) + yc))
                    t += .1
                positions_df = pd.DataFrame(positions,columns=(['x','y']))
                #print(positions_df)
                #pltFit.title(f'Set {set_number} Fitting a circle. Diameter = {2*r}')
                #pltFit.axis('equal')
            else:
                positions_df = pd.DataFrame(positions,columns=(['x','y']))


            
            plots[sheet_name][0].append(section_df)   #idx 0
            plots[sheet_name][1].append(circle_section_df) #idx 1
            plots[sheet_name][2].append(positions_df) #idx 2
            
            #df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            '''results = pd.concat([results, pd.DataFrame([{'SheetName':sheet_name, 'Start Index':section_df_start_index, 
                                    'End Index':section_df_end_index,'Set Number':set_number, 'Time':time_slots[i],
                                    'Lowest Value':row,'diameter':2*r,'Max X':circle_section_df['X'].min(), 
                                    'Cp':section_df['Cp'].iloc[0]}])], ignore_index=True)'''
            # for no cp and X
            results = pd.concat([results, pd.DataFrame([{'SheetName':sheet_name, 'Start Index':section_df_start_index, 
                                    'End Index':section_df_end_index,'Set Number':set_number, 'Time':time_slots[i],
                                    'Lowest Value':row,'diameter':2*r, 'Max X':circle_section_df['X'].min()}])], ignore_index=True)

        
            #if (first_run == 1):
            #    break;



        #Graphs per sheet including all sections
        plt.figure(figsize=(15,10))
        plt.xlabel('R (Ohm)')
        plt.ylabel('-Z (Ohm)')
        plt.grid()
        plt.title(f'{file_name} Sheet Name {sheet_name} R vs -Z plots per section')
        for plotNum in range(0,len(plots[sheet_name][0])):
            plt.plot(plots[sheet_name][0][plotNum]['Rs'], -plots[sheet_name][0][plotNum]['X'], '-', label=f'T {plotNum*1.5}')
        plt.legend(loc='upper left')
        pdfFile.savefig(dpi= 500)
        plt.show()
        plt.close()

        if notAsemicircle == 0:
            plt.figure(figsize=(15,10))
            plt.xlabel('R (Ohm)')
            plt.ylabel('-Z (Ohm)')
            plt.grid()
            plt.title(f'{file_name} Sheet Name {sheet_name} R vs -Z plots per section (only the curve part)')
            for plotNum in range(0,len(plots[sheet_name][1])):
                plt.plot(plots[sheet_name][1][plotNum]['Rs'], -plots[sheet_name][1][plotNum]['X'], '-', label=f'T {plotNum*1.5}')
                plt.legend(loc='upper left')
            pdfFile.savefig()
            plt.show()
            plt.close()

        if notAsemicircle == 0:
            plt.figure(figsize=(30,30))
            plt.xlabel('R (Ohm)')
            plt.ylabel('-Z (Ohm)')
            plt.xlim(0,30000)
            plt.ylim(-15000,15000)
            #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid()
            plt.title(f'{file_name} Sheet Name {sheet_name} R vs -Z plots per section (curve part fitted to circle)')
            for plotNum in range(0,len(plots[sheet_name][1])):
            
                plt.plot(plots[sheet_name][2][plotNum]['x'], -plots[sheet_name][2][plotNum]['y'], '-', color="yellow")
                plt.plot(plots[sheet_name][1][plotNum]['Rs'], -plots[sheet_name][1][plotNum]['X'], '-', color="blue")
            pdfFile.savefig()
            plt.show()
            plt.close()

        # Plotting the results
        plt.figure(figsize=(15,10))
        plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['diameter'], marker='o', linestyle='-', label=sheet_name)
        plt.xlabel('Time')
        plt.ylabel('RCT')
        plt.title(f'{file_name} {sheet_name} : RCT vs time')
        #plt.ylim(1000, 6000)
        plt.legend(loc='upper left')
        plt.grid(True)
        pdfFile.savefig()
        # Close the plot to free memory and avoid overlap
        plt.close()


        # Combined Max X plot for all sheets
        plt.figure(figsize=(15,10))
        plt.xlabel('Time')
        plt.ylabel('Max X')
        plt.title(f'{file_name} {sheet_name} : Max X vs time')
        plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['Max X'], marker='o', linestyle='-', label=sheet_name)
        plt.legend(loc='upper left')
        plt.grid(True)
        pdfFile.savefig()
        plt.show()
        plt.close()

        # Combined CP plot for all sheets
        plt.figure(figsize=(15,10))
        plt.xlabel('Time')
        plt.ylabel('Cp')
        plt.title(f'{file_name} {sheet_name} : Cp vs time')
        plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['Cp'], marker='o', linestyle='-', label=sheet_name)
        plt.legend(loc='upper left')
        plt.grid(True)
        pdfFile.savefig()
        plt.show()
        plt.close()
        

        # Plot the last section
        last_section_df = df.iloc[maxFreqIndices[-2] + 1:maxFreqIndices[-1] + 1]

        # Store the last section for the combined plot
        last_sections.append((sheet_name, last_section_df))

    # Combined all sheet and all sections

    def plotting(): 
        # Combined RCT plot for all sheets
        plt.figure(figsize=(15,10))
        plt.xlabel('Time')
        plt.ylabel('RCT')
        plt.title(f'{file_name} RCT vs time')
        for sheet_name in selected_sheets:
            plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['diameter'], marker='o', linestyle='-', label=sheet_name)
        plt.legend(loc='upper right')
        plt.grid(True)
        pdfFile.savefig()
        plt.show()
        plt.close()

        # Combined Max X plot for all sheets
        plt.figure(figsize=(15,10))
        plt.xlabel('Time')
        plt.ylabel('Max X')
        plt.title(f'{file_name} Max X vs time')

        for sheet_name in selected_sheets:
            plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['Max X'], marker='o', linestyle='-', label=sheet_name)
        plt.legend(loc='upper left')
        plt.grid()
        pdfFile.savefig()
        plt.show()
        plt.close()

        # Combined CP plot for all sheets
        plt.figure(figsize=(15,10))
        plt.xlabel('Time')
        plt.ylabel('Cp')
        plt.title(f'{file_name} Cp vs time')

        for sheet_name in selected_sheets:
            plt.plot(results[results['SheetName'] == sheet_name]['Time'], results[results['SheetName'] == sheet_name]['Cp'], marker='o', linestyle='-', label=sheet_name)
        plt.legend(loc='upper left')
        plt.grid()
        pdfFile.savefig()
        plt.show()
        plt.close()


        # Combined plot for the last sections of all sheets
        plt.figure(figsize=(15, 10))

        # Set combined plot labels and title
        plt.xlabel('R (Ohm)')
        plt.ylabel('-Z (Ohm)')
        plt.title('Last Sections - Combined Plot')

        plt.grid()
        for sheet_name, section_df in last_sections:
            plt.plot(section_df['Rs'], -section_df['X'], '-', label=f'Last section {sheet_name}')

        plt.legend(loc='upper left')
        pdfFile.savefig()
        plt.show()



        # circle portion of last section
        plt.figure(figsize=(15,10))
        plt.xlabel('R (Ohm)')
        plt.ylabel('-Z (Ohm)')
        plt.title('Last Sections - Combined Plot - Circle portions')

        plt.grid()
        for sheet_name, section_df in last_sections:
            plt.plot(plots[sheet_name][2][len(plots[sheet_name][2])-1]['x'], plots[sheet_name][2][len(plots[sheet_name][2])-1]['y'], '-', color="yellow")
            plt.plot(plots[sheet_name][1][len(plots[sheet_name][1])-1]['Rs'], plots[sheet_name][1][len(plots[sheet_name][1])-1]['X'], '-', color="blue")
        plt.legend(loc='upper left')
        pdfFile.savefig()
        plt.show()
        pdfFile.close()


        # Write results into csv
        results.to_csv(f'results_{file_name}.csv')

    plot_boolean = False
    if plot_boolean == True:
        plotting


def deepta_analysis_fucntions(df, cycle_idx, time_per_cycle, cp1, ph1, freq_array, Z_array):
    results = {
        "time(mins)": time_per_cycle * cycle_idx,
        "Cp1": cp1,
        "Ph1": ph1,
        "Rct_semicircle": None,
        "Rs": None,
        "fit_success": False
    }

    # Convert to float and remove NaNs/Infs
    x_raw = pd.to_numeric(df["Rs"], errors="coerce").to_numpy()
    y_raw = pd.to_numeric(df["X"], errors="coerce").to_numpy()
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x_raw = x_raw[mask]
    y_raw = np.abs(y_raw[mask])

    # Require at least 3 valid points
    if len(x_raw) < 3:
        return results  # skip cycle

    # Get region after minimum
    idx_min_y = np.argmin(y_raw)
    x_circle = x_raw[idx_min_y:]
    y_circle = y_raw[idx_min_y:]

    if len(x_circle) < 3:
        return results  # skip cycle

    try:
        circle_coords = np.column_stack((x_circle, y_circle))
        xc, yc, r, sigma = taubinSVD(circle_coords)

        results["Rct_semicircle"] = 2 * r      # Diameter ≈ Rct
        results["Rs"] = xc - r                 # Left intercept estimate
        results["fit_success"] = True

    except Exception:
        # failed fit → return default results (None values)
        return results

    return results


