
import pandas as pd
import sys

def filter_excel_by_filenames(file_list_path, input_excel_path, output_excel_path):
    # Read filenames from the text file
    with open(file_list_path, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]

    # Read the input Excel sheet
    try:
        excel_data = pd.read_excel(input_excel_path)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        sys.exit(1)

    # Filter the rows that match the filenames in the first column (or whichever column)
    # Assuming the filenames are in the first column (column 0)
    filtered_data = excel_data[excel_data.iloc[:, 2].isin(filenames)]

    # Write the filtered data to a new Excel sheet
    try:
        filtered_data.to_excel(output_excel_path, index=False)
        print(f"Filtered data saved to {output_excel_path}")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")
        sys.exit(1)

    # Assuming the filenames are in the first column (change '0' to the correct column index if necessary)
    excel_filenames = excel_data.iloc[:, 2].tolist()

    # Find the filenames that are not in the Excel sheet
    missing_filenames = [filename for filename in filenames if filename not in excel_filenames]

    # Print out the filenames that couldn't be found
    if missing_filenames:
        print("The following filenames were not found in the Excel sheet:")
        for missing in missing_filenames:
            print(missing)
    else:
        print("All filenames were found in the Excel sheet.")


if __name__ == "__main__":
    # Ensure correct usage
    if len(sys.argv) != 4:
        print("Usage: python filter_excel.py <file_list.txt> <input_excel.xlsx> <output_excel.xlsx>")
        sys.exit(1)

    # Assign command line arguments
    file_list_path = sys.argv[1]
    input_excel_path = sys.argv[2]
    output_excel_path = sys.argv[3]

    # Call the function to filter the Excel data based on filenames
    filter_excel_by_filenames(file_list_path, input_excel_path, output_excel_path)
