import os
import pandas as pd

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_acc补充"
    all_dfs = []

    # 1) Traverse all subfolders
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Skip non-folder items

        # 2) Look for "WEAR_ACC_ABSOLUTE_02.xlsx" in each subfolder
        acc_file = os.path.join(folder_path, "WEAR_ACC_ABSOLUTE_02.xlsx")
        if not os.path.exists(acc_file):
            continue  # Skip if the file does not exist

        # 3) Read the Excel file
        try:
            df = pd.read_excel(acc_file)
        except Exception as e:
            print(f"[ERROR] Failed to read file: {acc_file}, reason: {e}")
            continue

        if df.empty:
            print(f"[INFO] File {acc_file} is empty, skipping")
            continue

        # 4) Store the read DataFrame in the list
        #    If you want to retain subfolder info in the final table, add a column
        df["source_folder"] = folder_name  # Mark which subfolder this row came from
        all_dfs.append(df)

    # Exit if no data was collected
    if not all_dfs:
        print("[INFO] No data collected. Exiting.")
        return

    # 5) Concatenate all DataFrames
    big_df = pd.concat(all_dfs, ignore_index=True)

    # 6) Sort by absolute_time_str
    #    Try converting absolute_time_str to datetime format
    #    If the format is "YYYY-MM-DD HH:MM:SS.ffffff", the following works
    #    Use errors="coerce" to avoid exceptions on bad rows
    big_df["abs_time_parsed"] = pd.to_datetime(big_df["absolute_time_str"],
                                               errors="coerce",
                                               format="%Y-%m-%d %H:%M:%S.%f")
    # If you're unsure about the format, remove the format argument to let pandas auto-detect:
    # big_df["abs_time_parsed"] = pd.to_datetime(big_df["absolute_time_str"], errors="coerce")

    # Warn if any parsing failed
    failed_count = big_df["abs_time_parsed"].isna().sum()
    if failed_count > 0:
        print(f"[WARNING] {failed_count} rows could not convert absolute_time_str to datetime. "
              f"These may appear at the beginning or end after sorting.")

    # 7) Sort the merged DataFrame
    big_df.sort_values(by="abs_time_parsed", inplace=True, ignore_index=True)

    # 8) Output to a new Excel file
    out_path = "/media/robert/4TB-SSD/watchped_dataset/merged_all_acc02_sorted.xlsx"
    big_df.to_excel(out_path, index=False)
    print(f"[INFO] Merged file generated: {out_path}")

if __name__ == "__main__":
    main()
