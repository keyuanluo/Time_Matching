import os
import pandas as pd
from datetime import datetime

def parse_folder_name(folder_name):
    """
    Example folder name: WEAR_27_01_2022_21_06_38
    Represents: day_month_year_hour_minute_second
    Returns a datetime object
    """
    try:
        prefix = "WEAR_"
        if not folder_name.startswith(prefix):
            # Does not start with WEAR_, return None
            return None

        parts = folder_name.replace(prefix, "").split("_")
        if len(parts) < 6:
            print(f"[Error] Invalid folder name format (fewer than 6 parts): {folder_name}")
            return None

        day, month, year, hour, minute, sec = map(int, parts[:6])
        return datetime(year, month, day, hour, minute, sec)
    except Exception as e:
        print(f"[Error] Failed to parse folder name: {folder_name} - {str(e)}")
        return None

def detect_delimiter(filepath):
    """
    Smartly detect delimiter from the first line:
      - If contains ';' then sep=';'
      - If contains '\t' then sep='\t'
      - Otherwise, use whitespace delimiter (delim_whitespace=True)
    """
    if not os.path.exists(filepath):
        return None, "File does not exist"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            first_line = f.readline()
    except Exception as e:
        return None, f"Failed to read first line: {e}"

    if ';' in first_line:
        return (';', None)
    elif '\t' in first_line:
        return ('\t', None)
    else:
        # No ';' or '\t', use whitespace delimiter
        return (r'\s+', None)

def format_dmyhmsms(ts):
    """
    Convert Timestamp to string: day_month_year_hour_minute_second_millisecond (3 digits)
    Example: 27_01_2022_21_06_38_020000
    """
    if pd.isna(ts):
        return None
    if not isinstance(ts, pd.Timestamp):
        return None

    day    = ts.day
    month  = ts.month
    year   = ts.year
    hour   = ts.hour
    minute = ts.minute
    second = ts.second
    micro  = ts.microsecond  # 0 ~ 999999
    ms = micro // 1000       # Milliseconds 0 ~ 999
    return f"{day:02d}_{month:02d}_{year}_{hour:02d}_{minute:02d}_{second:02d}_{micro:06d}"

def process_gyro_file(folder_path, folder_time):
    """
    1) Smartly detect delimiter
    2) Read CSV (header in first row)
    3) Compute absolute_time (datetime64[ns])
    4) Generate absolute_time_str (microsecond precision)
    5) Generate formatted_time (day_month_year_hour_minute_second_millisecond)
    6) Output to Excel
    For the WEAR_GYRO.csv file
    """
    gyro_file = os.path.join(folder_path, "WEAR_GYRO.csv")
    if not os.path.exists(gyro_file):
        print(f"[Error] File not found: {gyro_file}")
        return

    # 1) Detect delimiter
    sep, err = detect_delimiter(gyro_file)
    if err:
        print(f"[Error] {err}")
        return
    if sep is None:
        print(f"[Error] Could not determine delimiter: {gyro_file}")
        return

    # 2) Read CSV
    try:
        if sep == r'\s+':
            df = pd.read_csv(gyro_file, delim_whitespace=True, header=0)
            print(f"[Debug] Read using whitespace delimiter: {gyro_file}")
        else:
            df = pd.read_csv(gyro_file, sep=sep, header=0)
            print(f"[Debug] Read using delimiter '{sep}': {gyro_file}")

        print("[Debug] Parsed column names:", df.columns.tolist())
        print("[Debug] First 3 rows:\n", df.head(3))

    except Exception as e:
        print(f"[Error] Failed to parse file: {gyro_file} - {str(e)}")
        return

    if df.empty:
        print(f"[Error] Empty file: {gyro_file}")
        return

    if 't' not in df.columns:
        print(f"[Error] Missing 't' column: {gyro_file}, Parsed columns: {df.columns.tolist()}")
        return

    # 3) Get first row's 't' (nanoseconds)
    t0_ns = df.iloc[0]['t']
    try:
        t0_ns = float(t0_ns)
    except:
        print(f"[Error] First row 't' is not a valid number: {gyro_file} -> {t0_ns}")
        return

    # 4) Compute absolute_time for each row
    base_time = pd.Timestamp(folder_time) - pd.to_timedelta(t0_ns, unit='ns')
    try:
        df['t'] = df['t'].astype(float)
    except:
        print(f"[Error] Column 't' cannot be converted to float: {gyro_file}")
        return

    df['absolute_time'] = base_time + pd.to_timedelta(df['t'], unit='ns')

    # 5) Create string column (microsecond precision)
    df['absolute_time_str'] = df['absolute_time'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # 6) Add new column: formatted_time
    df['formatted_time'] = df['absolute_time'].apply(format_dmyhmsms)

    # Output to Excel
    output_file = os.path.join(folder_path, "WEAR_GYRO_ABSOLUTE_01.xlsx")
    try:
        df.to_excel(output_file, index=False)
        print(f"[Info] File generated: {output_file}")
    except Exception as e:
        print(f"[Error] Failed to write Excel: {output_file} - {str(e)}")

def main():
    """
    Traverse all subfolders under /media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充,
    For folders starting with WEAR_, process them using process_gyro_file.
    """
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"

    # Traverse all subfolders
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Only process folders

        # Parse folder name -> datetime
        folder_time = parse_folder_name(folder_name)
        if not folder_time:
            # Parsing failed or not matching WEAR_ naming, skip
            continue

        # Process GYRO file
        process_gyro_file(folder_path, folder_time)

if __name__ == "__main__":
    main()
