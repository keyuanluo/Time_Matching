import os
import re
import pandas as pd

def extract_video_num(video_id):
    """
    Assuming the format of video_id is something like "video_0001", "video_0010", etc.,
    extract the number part and convert it to int.
    """
    # Method 1: Simple split
    # parts = video_id.split("_")  # ["video", "0001"]
    # return int(parts[1])
    #
    # Method 2: Use regex to extract digits
    match = re.search(r"(\d+)$", video_id)
    if match:
        return int(match.group(1))
    else:
        # If no match is found, return a large number or handle differently as needed
        return 999999999

def main():
    # Input Excel file
    in_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched_02.xlsx"
    # Output Excel file
    out_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched_03.xlsx"

    # Read the original Excel file
    df = pd.read_excel(in_excel)

    # The following columns are required; make sure the names match:
    # df.columns should include at least:
    #   ["video_id", "image_name", "image_original_time", "image_formatted_time", ...]
    #   as well as "acc_formatted_time", "acc_x", "acc_y", "acc_z", "time_diff_s", etc.
    # Extra columns (if any) do not need to be removed

    # 1) Extract the numeric part of video_id
    df["video_num"] = df["video_id"].apply(extract_video_num)

    # 2) Convert image_name to int for sorting
    # Note: If image_name is already numeric strings and contains no non-numeric values, this will work
    # If non-numeric values exist, you need to clean or handle exceptions first
    df["image_int"] = df["image_name"].astype(int)

    # 3) Sort by video_num (ascending) + image_int (ascending)
    df.sort_values(by=["video_num", "image_int"], ascending=[True, True], inplace=True)

    # 4) Reformat image_name to 6-digit strings
    #    e.g., 1 -> "000001", 15 -> "000015"
    df["image_name"] = df["image_int"].apply(lambda x: f"{x:06d}")

    # 5) Drop intermediate columns if no longer needed
    df.drop(columns=["video_num", "image_int"], inplace=True)

    # 6) Save the result
    df.to_excel(out_excel, index=False)
    print(f"[INFO] New file generated: {out_excel}")

if __name__ == "__main__":
    main()
