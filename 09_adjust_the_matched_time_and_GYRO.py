import os
import re
import pandas as pd

def extract_video_num(video_id):
    """
    Assuming the format of video_id is something like "video_0001", "video_0010", etc.,
    extract the number part and convert it to an integer.
    """
    match = re.search(r"(\d+)$", video_id)
    if match:
        return int(match.group(1))
    else:
        # If no match is found, return a large number or handle as needed
        return 999999999

def main():
    # Input file (GYRO version)
    in_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_02.xlsx"
    # Output file
    out_excel = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_gyro_matched_03.xlsx"

    # Read the original Excel file
    df = pd.read_excel(in_excel)

    # 1) Extract video_num
    df["video_num"] = df["video_id"].apply(extract_video_num)

    # 2) Convert image_name to int for sorting
    df["image_int"] = df["image_name"].astype(int)

    # 3) Sort by video_num (ascending) + image_int (ascending)
    df.sort_values(by=["video_num", "image_int"], ascending=[True, True], inplace=True)

    # 4) Reformat image_name as a 6-digit string (e.g. 1 -> 000001)
    df["image_name"] = df["image_int"].apply(lambda x: f"{x:06d}")

    # 5) Drop intermediate columns if no longer needed
    df.drop(columns=["video_num", "image_int"], inplace=True)

    # 6) Save the result
    df.to_excel(out_excel, index=False)
    print(f"[INFO] New file generated: {out_excel}")

if __name__ == "__main__":
    main()
