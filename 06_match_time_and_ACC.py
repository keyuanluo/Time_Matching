import os
import bisect
import pandas as pd
import datetime

def parse_acc_formatted_time(timestr):
    """
    Parse the formatted_time from the ACC file (e.g. "27_01_2022_19_35_28_000000")
    into a Python datetime object. Return None if parsing fails.
    Format: day_month_year_hour_minute_second_microsecond
    """
    parts = timestr.split('_')
    if len(parts) != 7:
        return None
    try:
        day    = int(parts[0])
        month  = int(parts[1])
        year   = int(parts[2])
        hour   = int(parts[3])
        minute = int(parts[4])
        second = int(parts[5])
        micro  = int(parts[6])
        dt = datetime.datetime(year, month, day, hour, minute, second, micro)
        return dt
    except:
        return None

def parse_image_formatted_time(timestr):
    """
    Parse formatted_time from video/image files (e.g. "06_01_2023_17_07_44_229000").
    Same format: day_month_year_hour_minute_second_microsecond.
    Reuses the same parsing logic as ACC.
    """
    return parse_acc_formatted_time(timestr)

def load_acc_data(acc_excel_path):
    """
    Read the ACC file `merged_all_acc02_sorted.xlsx`, and extract:
      - formatted_time
      - x, y, z
    Then parse formatted_time to datetime and sort by time.
    Returns a list of:
    [ (dt, x, y, z, formatted_time_str), ... ]
    """
    df = pd.read_excel(acc_excel_path)

    # Ensure required columns exist
    for col in ["formatted_time", "x", "y", "z"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in ACC file")

    acc_list = []
    for _, row in df.iterrows():
        ft_str = str(row["formatted_time"])
        dt = parse_acc_formatted_time(ft_str)
        if dt is None:
            continue  # Skip if parsing fails
        x_val = row["x"]
        y_val = row["y"]
        z_val = row["z"]
        acc_list.append((dt, x_val, y_val, z_val, ft_str))

    # Sort by datetime
    acc_list.sort(key=lambda r: r[0])
    return acc_list

def find_nearest_acc_time(acc_list, dt_query):
    """
    Given a sorted acc_list: [(dt, x, y, z, ft_str), ...]
    and a query datetime `dt_query`, use binary search to find
    the nearest timestamp. Returns:
    (dt_acc, x, y, z, formatted_time, diff_seconds)
    Returns None if acc_list is empty.
    """
    if not acc_list:
        return None

    acc_dts = [item[0] for item in acc_list]
    idx = bisect.bisect_left(acc_dts, dt_query)

    candidates = []
    if idx > 0:
        candidates.append(idx - 1)
    if idx < len(acc_list):
        candidates.append(idx)

    best = None
    best_diff = None
    for c in candidates:
        if c < 0 or c >= len(acc_list):
            continue
        dt_acc, x, y, z, ft_str = acc_list[c]
        diff = abs((dt_acc - dt_query).total_seconds())
        if (best is None) or (diff < best_diff):
            best = (dt_acc, x, y, z, ft_str)
            best_diff = diff

    if best is None:
        return None
    dt_acc, x, y, z, ft_str = best
    return (dt_acc, x, y, z, ft_str, best_diff)

def merge_video_image_with_acc(acc_excel_path, video_image_excel_path, out_excel_path):
    """
    Main function:
    1. Load and parse ACC data
    2. Load and parse video/image data
    3. For each video/image row, find the nearest ACC time using image_formatted_time
    4. Merge and output to out_excel_path
    """
    print("[INFO] Loading ACC data...")
    acc_list = load_acc_data(acc_excel_path)
    print(f"[INFO] Loaded {len(acc_list)} ACC entries")

    print("[INFO] Loading video/image data...")
    df_video = pd.read_excel(video_image_excel_path)
    needed_cols = ["video_id", "image_name", "image_original_time", "image_formatted_time"]
    for c in needed_cols:
        if c not in df_video.columns:
            raise ValueError(f"Missing column '{c}' in video/image file")

    matched_acc_time = []
    matched_x = []
    matched_y = []
    matched_z = []
    matched_diff_sec = []

    print("[INFO] Starting matching process...")
    for idx, row in df_video.iterrows():
        img_ft_str = str(row["image_formatted_time"])
        dt_img = parse_image_formatted_time(img_ft_str)
        if dt_img is None:
            # Fill with empty or NaN if parsing fails
            matched_acc_time.append("")
            matched_x.append(float("nan"))
            matched_y.append(float("nan"))
            matched_z.append(float("nan"))
            matched_diff_sec.append(float("nan"))
            continue

        res = find_nearest_acc_time(acc_list, dt_img)
        if res is None:
            # No ACC data available
            matched_acc_time.append("")
            matched_x.append(float("nan"))
            matched_y.append(float("nan"))
            matched_z.append(float("nan"))
            matched_diff_sec.append(float("nan"))
        else:
            dt_acc, x_val, y_val, z_val, ft_acc, diff_s = res
            matched_acc_time.append(ft_acc)
            matched_x.append(x_val)
            matched_y.append(y_val)
            matched_z.append(z_val)
            matched_diff_sec.append(diff_s)

    # Append to df_video
    df_video["acc_formatted_time"] = matched_acc_time
    df_video["acc_x"] = matched_x
    df_video["acc_y"] = matched_y
    df_video["acc_z"] = matched_z
    df_video["time_diff_s"] = matched_diff_sec

    print(f"[INFO] Matching complete. Total rows: {len(df_video)}. Writing to: {out_excel_path}")
    df_video.to_excel(out_excel_path, index=False)
    print("[DONE]")

if __name__ == "__main__":
    # Example file paths
    acc_file_path = "/media/robert/4TB-SSD/watchped_dataset/merged_all_acc02_sorted.xlsx"
    video_file_path = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_time_fixed.xlsx"
    output_file_path = "/media/robert/4TB-SSD/watchped_dataset/combined_video_image_and_acc_matched.xlsx"

    merge_video_image_with_acc(
        acc_excel_path=acc_file_path,
        video_image_excel_path=video_file_path,
        out_excel_path=output_file_path
    )
