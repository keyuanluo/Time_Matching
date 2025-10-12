import os
import pandas as pd

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc"
    all_dfs = []

    # 1) Traverse all subfolders
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip non-directory items

        # 2) Look for "WEAR_GYRO_ABSOLUTE_02.xlsx" in each subfolder
        gyro_file = os.path.join(folder_path, "WEAR_GYRO_ABSOLUTE_02.xlsx")
        if not os.path.exists(gyro_file):
            continue  # Skip if the file doesn't exist

        # 3) Read the Excel file
        try:
            df = pd.read_excel(gyro_file)
        except Exception as e:
            print(f"[ERROR] Failed to read file: {gyro_file}, reason: {e}")
            continue

        if df.empty:
            print(f"[INFO] File {gyro_file} is empty, skipping")
            continue

        # 4) Store the DataFrame in the list
        #    If you want to keep track of the source folder, add a column
        df["source_folder"] = folder_name  # Tag the row with its originating folder
        all_dfs.append(df)

    # If no data collected, exit
    if not all_dfs:
        print("[INFO] No data collected. Exiting.")
        return

    # 5) Merge all DataFrames
    big_df = pd.concat(all_dfs, ignore_index=True)

    # 6) Sort based on absolute_time_str
    #    Try converting absolute_time_str to datetime format
    big_df["abs_time_parsed"] = pd.to_datetime(big_df["absolute_time_str"],
                                               errors="coerce",
                                               format="%Y-%m-%d %H:%M:%S.%f")

    # Warn if any rows failed to parse
    failed_count = big_df["abs_time_parsed"].isna().sum()
    if failed_count > 0:
        print(f"[WARNING] {failed_count} rows could not convert 'absolute_time_str' to datetime. "
              f"They may appear at the beginning or end after sorting.")

    # 7) Sort the merged table
    big_df.sort_values(by="abs_time_parsed", inplace=True, ignore_index=True)

    # 8) Output to a new Excel file
    out_path = "/media/robert/4TB-SSD/watchped_dataset/merged_all_gyro02_sorted.xlsx"
    big_df.to_excel(out_path, index=False)
    print(f"[INFO] Merged file created: {out_path}")

if __name__ == "__main__":
    main()
