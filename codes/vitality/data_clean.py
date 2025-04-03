import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


###################################################################
###                    Notebook 1. Section 3                    ###
###################################################################

### Load & Process CSV Files ###

def find_matching_csvs(in_dir, var):
    """
    Searches for CSV files in a directory that contain the exact keyword (var)
    in their filenames (case-insensitive match on split words).

    Parameters:
        in_dir (str): Directory to search in.
        var (str): Keyword to match exactly in filenames.

    Returns:
        list: Sorted list of matching file paths.
    """
    file_paths = sorted([
        os.path.join(in_dir, f)
        for f in os.listdir(in_dir)
        if f.endswith('.csv') and any(
            part.lower() == var.lower() 
            for part in f.replace('.', ' ').replace('_', ' ').split()
        )
    ])
    return file_paths


def load_and_process_csv_files(in_dir, var):
    """
    Load data from matched CSV files, clean the data, and return as a list.
    
    Parameters:
    - in_dir (str): Directory path containing the CSV files.
    - var (str): Exact keyword to search for in the filenames.
    
    Returns:
    - data (list): List of cleaned DataFrames.
    """
    
    file_paths = find_matching_csvs(in_dir, var)

    # Step 2: Load and clean CSV files
    data = []
    for file in file_paths:
        df = pd.read_csv(file, low_memory=False)

        # Rename "Time" column to "datetime" and clean up its values
        if 'Time' in df.columns:
            df.rename(columns={'Time': 'datetime'}, inplace=True)
            df['datetime'] = df['datetime'].str.split(' ').str[:2].str.join(' ')

        # Simplify other column names to words in parentheses, converted to lowercase
        df.rename(
            columns={
                col: col[col.find('(') + 1:col.find(')')].lower() 
                for col in df.columns 
                if '(' in col and ')' in col
            },
            inplace=True
        )

        # Append the cleaned DataFrame to the list
        data.append(df)

    # Output information
    print(f"Number of '{var}' files processed: {len(data)}")
    print("Files loaded:")
    for file_path in file_paths:
        print(f"{file_path}")
    
    return data



###################################################################
###                    Notebook 1. Section 4                    ###
###################################################################

### Merge DataFrames Horizontally ###

def merge_multiple_dfs(df_list, freq):
    
    """
    Merge multiple DataFrames with shared datetime columns into one, 
    aligned to a complete time range at the specified frequency.

    Parameters:
        df_list (list of pd.DataFrame): List of DataFrames, each with a 'datetime' column.
        freq (str): Frequency of time stamps (e.g., "15min").
    
    Returns:
        pd.DataFrame: A single DataFrame with all merged data on a uniform time index.
    """
    
    # Step 1: Convert datetime column to datetime64[ns] format and drop duplicates
    cleaned_dfs = []
    for df in df_list:
        df['datetime'] = pd.to_datetime(df['datetime'])
        cleaned_df = df.drop_duplicates(subset='datetime').reset_index(drop=True)
        cleaned_dfs.append(cleaned_df)
    
    # Step 2: Determine the full time range
    all_datetimes = pd.concat([df['datetime'] for df in cleaned_dfs])
    start, end = all_datetimes.min(), all_datetimes.max()
    
    # Create the full range of timestamps
    full_timestamps = pd.date_range(start=start, end=end, freq=freq)
    
    # Step 3: Merge all DataFrames with full timestamps
    # Create a base DataFrame with the full timestamps
    merged_df = pd.DataFrame({'datetime': full_timestamps})
    
    for df in cleaned_dfs:
        # Merge with the current DataFrame on 'datetime'
        merged_df = merged_df.merge(df, on='datetime', how='left')
    
    # drop the last row: first timestamp (00:00:00) of the next day
    merged_df = merged_df.drop(merged_df.index[-1])
    
    return merged_df



###################################################################
###                    Notebook 1. Section 5                    ###
###################################################################

### Concatenate DataFrames Vertically ###

def concat_dfs(in_dir, var):
    """
    Load data from matched CSV files, aligns their column order, and combines them into a single DataFrame.

    Parameters:
        in_dir (str): Directory to search for CSV files.
        var (str): Keyword to match exactly (case-insensitive) in file names.

    Returns:
        pd.DataFrame: Combined DataFrame with consistent columns and unique datetime entries.
    """
    
    file_paths = find_matching_csvs(in_dir, var)

    # Step 2: Load those CSV files into a list
    dfs = [pd.read_csv(file, low_memory=False) for file in file_paths]

    if not dfs:
        return pd.DataFrame()  # Return empty DataFrame if no files matched

    # Align column order of all DataFrames in dfs with the first one
    dfs = [df[dfs[0].columns] for df in dfs]

    # Step 3: Combine all DataFrames and drop duplicate timestamps, if any
    df_combined = pd.concat(dfs).drop_duplicates(subset="datetime").reset_index(drop=True)

    # Convert datetime column to datetime64[ns] format
    df_combined["datetime"] = pd.to_datetime(df_combined["datetime"])

    return df_combined



###################################################################
###                 Notebook 1. Section 4 & 5                   ###
###################################################################

### Save File ###

def save_file(data, fname, dname):
    """
    Save a data file (data) to a specific location (dname) and filename (fname).
    
    Automatically overwrites any existing CSV file with the same name.
    """
    # Ensure the directory exists
    if not os.path.exists(dname):
        os.mkdir(dname)
        print(f'Directory {dname} was created.')
        
    fpath = os.path.join(dname, fname)
    
    # Automatically overwrite the file
    print(f'Writing file: "{fpath}"')
    data.to_csv(fpath, index=False)



###################################################################
###                   Notebook 2. Section 1.3                   ###
###################################################################

### Plot Data ###

def plot_data(df, var, txt="original", fig_path="./"):
    """
    Plots the data with 'datetime' on the x-axis and other columns as separate lines and saves the figure to a PDF file.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to plot, including a 'datetime' column.
    - var (str): Label for the y-axis, representing the variable being plotted.
    - txt (str, optional): Additional text for the plot title. Defaults to 'original'.
    - fig_path (str, optional): Directory path where the PDF file will be saved. Defaults to './'.
    """

    # Ensure 'datetime' is the DataFrame's index
    if df.index.name != 'datetime':
        df = df.set_index('datetime')

    # Plot the DataFrame
    df.plot(figsize=(10, 6), alpha=0.9)

    # Set plot labels and title
    plt.xlabel('datetime', fontsize=13)
    plt.ylabel(var, fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Meter Name', loc='upper left', fontsize=11, title_fontsize=12)
    plt.title(f'{var} ({txt})', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Ensure the directory exists
    os.makedirs(fig_path, exist_ok=True)

    # Save the figure to a PDF file
    pdf_filename = os.path.join(fig_path, f"{var}_{txt}.pdf")
    plt.savefig(pdf_filename, format='pdf')
    print(f"Plot saved to {pdf_filename}")

    # Display the plot
    plt.show()



### Reshape Dataframe for Original Data ###

def reshape_merged_df(df):
    
    """
    Reshape a merged DataFrame by converting all columns except 'datetime' into two columns: 'meter_name' and 'meter_reading'.

    Parameters:
        df (pd.DataFrame): The merged DataFrame with a 'datetime' column and other meter columns.
    
    Returns:
        pd.DataFrame: Reshaped DataFrame with columns ['datetime', 'meter_reading', 'meter_name'].
    """
    
    # Ensure the 'datetime' column is of datetime type, and print out the datetime range for verification
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Melt the DataFrame to reshape
    reshaped_df = pd.melt(
        df,
        id_vars=['datetime'],            # Keep 'datetime' column
        var_name='meter_name',           # New column for former column names
        value_name='meter_reading'       # New column for values
    )
    
    # Sort by 'datetime' and 'meter_name'
    reshaped_df = reshaped_df.sort_values(by=['datetime', 'meter_name']).reset_index(drop=True)
    
    # Ensure the 'datetime' column remains in the correct datetime format
    reshaped_df['datetime'] = pd.to_datetime(reshaped_df['datetime'])
  
    return reshaped_df



###################################################################
###                  Notebook 2. Section 2.1                    ###
###################################################################

### Handle Meter Issues ###

def fix_meter_issues(var, filepath, df):
    """
    Fix meter issues based on the log file.

    Parameters:
    var (str): The variable to filter the log (e.g., 'kwh').
    filepath (str): The path to the Excel file with the issue log.
    df (pd.DataFrame): The dataframe containing meter readings.

    Returns:
    pd.DataFrame: The updated dataframe with meter issues fixed.
    """
    # Create a copy of the dataframe to avoid modifying the original
    fixed_df = df.copy()

    # Read the meter issue log
    log = pd.read_excel(filepath, sheet_name="meter_issue_log")

    # Filter rows based on the specified variable (case-insensitive match)
    # var_filtered_log = log[log['variable'].str.contains(fr'\b{var}\b', na=False, case=False)]
    var_filtered_log = log[log['variable'].apply(lambda x: var in [v.strip() for v in x.split(',')])]

    # Iterate over the filtered rows to fix issues
    for _, row in var_filtered_log.iterrows():
        meter_names = row['meter_name']
        start_datetime = row['start_datetime']
        end_datetime = row['end_datetime']
        imputed_value = row['imputed_value']

        # Skip rows with missing meter_name
        if pd.isna(meter_names):
            continue
            
        # Split and strip meter names if multiple meters are listed
        meter_list = [meter.strip() for meter in meter_names.split(',')]

        # Convert start and end datetimes to pandas datetime
        start_datetime = pd.to_datetime(start_datetime, errors='coerce')
        end_datetime = pd.to_datetime(end_datetime, errors='coerce')

        # If start_datetime is NaT, set the start time to the first available datetime in the dataframe
        if pd.isna(start_datetime):
            start_datetime = fixed_df['datetime'].min()

        # If end_datetime is NaT, set the end time to the last available datetime in the dataframe
        if pd.isna(end_datetime):
            end_datetime = fixed_df['datetime'].max()

        # Mask for the datetime range
        datetime_mask = (fixed_df['datetime'] >= start_datetime) & (fixed_df['datetime'] <= end_datetime)

        # Replace values based on the var condition
        for meter in meter_list:
            if meter in fixed_df.columns:
                if var in ["kwh", "tonhrs"]:
                    # Use imputed_value for replacement
                    if imputed_value == "NaN":
                        imputed_value = np.nan
                    fixed_df.loc[datetime_mask, meter] = imputed_value
                else:
                    # Set to NaN if var is not "kwh" or "tonhrs"
                    fixed_df.loc[datetime_mask, meter] = np.nan

    return fixed_df



###################################################################
###                  Notebook 2. Section 2.2                    ###
###################################################################

### Data Cleaning and Linear Interpolation ###

def clean_and_interpolate(column, window, flags):
    """
    Clean data errors and do linear interpolation.
    
    Input Args:
    column: a column in a pandas dataframe
    window: integer, (estimated) maximum number of consecutive sudden drops
    flags: a corresponding column tracking the interpolated values
    
    Output Returns:
    column: updated column with interpolated values    
    Note: flags would also be updated, value=1 means the value in the updated column is interpolated
    """
        
    column_cp = column.copy()  # Copy the original column
    
    # STEP 1: Handle sudden drops (non-monotonic decreases)
    for i in range(1, len(column)):
        # Find the first non-NaN value before index i
        non_nan_index_before = column[:i].last_valid_index()

        # Adjust window dynamically for the last points
        max_check_range = min(len(column) - i, window)

        # Check if a valid index was found and compare values
        if non_nan_index_before is not None and column[i] < column[non_nan_index_before] \
           and (column[i + 1:i + 1 + max_check_range] >= column[non_nan_index_before]).any():
            column[i] = np.nan  # Set to NaN for further interpolation

    # STEP 2: Handle stuck points (consecutive identical values, except 0)
    mask = (column.duplicated(keep='first')) & (column != 0.)  
    column = column.mask(mask)  # Replace with NaN
    
    # STEP 3: Linear interpolation for all remaining NaNs in the middle
    valid_start = column.first_valid_index()  # Get the first valid index
    valid_end = column.last_valid_index()  # Get the last valid index
    
    # Ensure valid_start and valid_end are within bounds before interpolating
    if valid_start is not None and valid_end is not None and valid_start < valid_end:
        # Interpolate in between valid_start and valid_end
        column[valid_start:valid_end] = column[valid_start:valid_end].interpolate(method='linear')

    # STEP 4: Record interpolation flags only for the values that have been changed (interpolated)
    # Only for values between `valid_start` and `valid_end` are checked
    if valid_start is not None and valid_end is not None:
        changed_mask = (column_cp[valid_start:valid_end] != column[valid_start:valid_end]) \
                       & column[valid_start:valid_end].notna()
        
        # Update only the relevant rows in `flags`
        flags.loc[valid_start:valid_end] = np.where(changed_mask, 1, flags.loc[valid_start:valid_end])
    
    return column



### Reshape Interpolated Data ###

def conditional_round(x):
    """
    Format numbers: 
    - If the value has more than 1 decimal, round to 2 decimals; otherwise, keep as-is
    """
    return round(x, 2) if (x * 10) % 10 != 0 else round(x, 1)  


def reshape_interpolated_data(df, interpolation_flags):
    """
    Reshape the dataframe and merge with interpolation_flags. 
    
    Input Args:
    df: a pandas dataframe
    interpolation_flags: a pandas dataframe with the same shape as df, containing is_interpolated flags of values in df
        
    Output Returns:
    final_df: combined and reshaped dataframe
    """
          
    # Step 1: Join df with interpolation_flags
    combined_df = pd.concat([df, interpolation_flags], axis=1)

    # Step 2: Reset the index (date range) and name it as 'datetime' to keep track
    combined_df = combined_df.reset_index().rename(columns={'index': 'datetime'})

    # Step 3: Reshape the meter readings into one column "meter_reading" and their names into "meter_name"
    meter_columns = df.columns.tolist()  # The original meter reading columns
    flag_columns = interpolation_flags.columns.tolist()  # The corresponding interpolation flag columns

    # Reshape meter readings into one column "meter_reading" and meter names into "meter_name"
    reshaped_df = pd.melt(combined_df, id_vars=['datetime'], 
                          value_vars=meter_columns, 
                          var_name='meter_name', 
                          value_name='meter_reading')

    # Reshape interpolation flags into one column "is_interpolated" and match with reshaped_df
    reshaped_flags = pd.melt(combined_df, id_vars=['datetime'], 
                             value_vars=flag_columns, 
                             var_name='meter_name_flag', 
                             value_name='is_interpolated')

    # Ensure flag names match meter names by removing '_interpolated' suffix
    reshaped_flags['meter_name'] = reshaped_flags['meter_name_flag'].str.replace('_interpolated', '')

    # Drop the extra 'meter_name_flag' column from reshaped_flags
    reshaped_flags = reshaped_flags.drop(columns=['meter_name_flag'])

    # Merge reshaped_df and reshaped_flags on 'datetime' and 'meter_name'
    final_df = pd.merge(reshaped_df, reshaped_flags, on=['datetime', 'meter_name'])

    # Apply conditional rounding to the 'meter_reading' column
    final_df['meter_reading'] = final_df['meter_reading'].apply(conditional_round)  

    # Sort the final dataframe by 'meter_name' and then 'datetime'
    final_df = final_df.sort_values(by=['datetime', 'meter_name']).reset_index(drop=True)
    
    # Return the final reshaped DataFrame
    return final_df



###################################################################
###                  Notebook 2. Section 3.1                    ###
###################################################################

### Calculate Interval ###

def calculate_delta_df(df, freq):
    """
    Calculate delta_df and Clean data.
    
    Input Args:
    df: a pandas dataframe.
    freq: time interval between consecutive data points (e.g., "15min", "1H").
        
    Output Returns:
    delta_df: cleaned delta_df (value at t minus value at t-1).
    """
    
    # STEP 1: Calculate delta_df
    delta_df = df.diff()

    # STEP 2: Handle spikes when the previous value in df is zero
    for col in delta_df.columns:
        # Identify spikes: current value is not zero, but the previous value is zero
        spike_mask = (df[col] != 0) & (df[col].shift(1) == 0)
        
        # Mark these spikes as zero in delta_df
        delta_df[col] = delta_df[col].where(~spike_mask, 0)
    
    # STEP 3: Handle negative delta_df values based on the following diff
    for col in delta_df.columns:
        neg_mask = delta_df[col] < 0  # Find negative values
        next_diff = delta_df[col].shift(-1)  # Look at the following diff
        
        # If the following diff is zero, set the negative diff to zero
        set_zero_mask = neg_mask & (next_diff == 0)
        delta_df[col] = delta_df[col].where(~set_zero_mask, 0)
        
        # If the following diff is not zero, mask the negative diff as NaN
        set_nan_mask = neg_mask & (next_diff != 0)
        delta_df[col] = delta_df[col].where(~set_nan_mask, np.nan)

        # Apply interpolation only between the first and last valid index
        valid_start = delta_df[col].first_valid_index()
        valid_end = delta_df[col].last_valid_index()

        if valid_start is not None and valid_end is not None:
            delta_df.loc[valid_start:valid_end, col] = delta_df.loc[valid_start:valid_end, col].interpolate(method='linear')

    # STEP 4: Calculate the daily count based on the frequency
    if isinstance(freq, str):
        count = int(pd.Timedelta('1D') / pd.Timedelta(freq))  # Number of timestamps per day
    else:
        raise ValueError("The 'freq' parameter must be a valid pandas frequency string (e.g., '15min').")

    # STEP 5: Handle zero periods by using the previous year’s same period value
    for col in delta_df.columns:
        zero_mask = delta_df[col] == 0
        for i in range(count * 365, len(delta_df)):  # Assuming 365 days (1 year) difference
            if zero_mask.iloc[i]:
                delta_df[col].iloc[i] = delta_df[col].iloc[i - count * 365]  # Use previous year's same period value.
                            
    return delta_df



### Reshape Interval Dataframe ###

def reshape_delta_df(df, var):
    
    """
    Reshape a merged DataFrame by converting all columns except 'datetime' into two columns: 'meter_name' and 'meter_reading'.

    Parameters:
        df (pd.DataFrame): The merged DataFrame with a 'datetime' column and other meter columns.
    
    Returns:
        pd.DataFrame: Reshaped DataFrame with columns ['datetime', 'meter_reading', 'meter_name'].
    """
    
    # Ensure the 'datetime' column is of datetime type, and print out the datetime range for verification
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Melt the DataFrame to reshape
    ValueName = 'delta_'+var             # New column name for values
    reshaped_delta_df = pd.melt(
        df,
        id_vars=['datetime'],            # Keep 'datetime' column
        var_name='meter_name',           # New column for former column names
        value_name=ValueName             # New column for values
    )
    
    # Sort by 'datetime' and 'meter_name'
    reshaped_delta_df = reshaped_delta_df.sort_values(by=['datetime', 'meter_name']).reset_index(drop=True)

    # Apply conditional rounding to the ValueName column
    reshaped_delta_df[ValueName] = reshaped_delta_df[ValueName].apply(conditional_round)     
    
    # Replace 0 value with NaN missing value
    reshaped_delta_df[ValueName] = reshaped_delta_df[ValueName].replace(0, np.nan)
    
    # Ensure the 'datetime' column remains in the correct datetime format
    reshaped_delta_df['datetime'] = pd.to_datetime(reshaped_delta_df['datetime'])
    
    return reshaped_delta_df