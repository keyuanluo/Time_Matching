import os
import pandas as pd
import datetime

# Global statistics
global_fix_count = 0
global_unfix_count = 0
unfixable_details = []  # Stores (video_id, row_idx, raw_time_str)

def load_video_size_map(width_height_excel):
    """
    Load video_width_height.xlsx and return a dictionary like:
    { 'video_0001': (1280, 720), ... }
    Assumes the Excel has three columns: ["video_id", "width", "height"].
    """
    df = pd.read_excel(width_height_excel)
    video2size = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])  # e.g. "video_0001"
        w = int(row["width"])
        h = int(row["height"])
        video2size[vid] = (w, h)
    return video2size

def parse_time_720(time_str):
    """
    Parse time string from 1280×720 resolution:
      Example: "1/6/23 17:07:44.229"
      => Returns datetime object (adjusts %y to full year 20yy)
    """
    dt = datetime.datetime.strptime(time_str, "%m/%d/%y %H:%M:%S.%f")
    if dt.year < 2000:
        dt = dt.replace(year=dt.year + 2000)
    return dt

def parse_time_1080(time_str):
    """
    Parse time string from 1920×1080 resolution:
      Example: "06/15/2022 20:33:53.090"
      => Returns datetime object
    """
    dt = datetime.datetime.strptime(time_str, "%m/%d/%Y %H:%M:%S.%f")
    return dt

def format_datetime(dt):
    """
    Convert datetime to format "day_month_year_hour_minute_second_microsecond_6digits"
    Example: 06_01_2023_17_44_11_000000
    """
    return dt.strftime("%d_%m_%Y_%H_%M_%S_%f")

def process_excel(video_id, in_excel, out_excel, width, height):
    """
    Read in_excel, determine how to parse the second column time based on (width, height),
    rename columns, generate third column image_formatted_time, and save to out_excel.

    If a row cannot be parsed and the previous row exists, add +0.033333 seconds;
    otherwise it is considered unfixable.

    Also pads image_name to 6 digits.
    """
    global global_fix_count, global_unfix_count, unfixable_details

    df = pd.read_excel(in_excel)
    if df.empty:
        print(f"[WARNING] File {in_excel} is empty. Skipping.")
        return

    old_cols = df.columns.tolist()
    if len(old_cols) < 2:
        print(f"[WARNING] File {in_excel} has fewer than 2 columns. Skipping.")
        return

    # Rename the first two columns
    df.rename(columns={
        old_cols[0]: "image_name",
        old_cols[1]: "image_original_time"
    }, inplace=True)

    # Keep only the first two columns
    df = df[["image_name", "image_original_time"]]

    # Pad image_name to 6 digits
    def pad_image_name(x):
        s = str(x).strip()
        try:
            num = int(float(s))  # Handles "1.0"
            return f"{num:06d}"
        except:
            return s  # If not a number, return as-is

    df["image_name"] = df["image_name"].apply(pad_image_name)

    # Choose parsing function based on resolution
    if (width, height) == (1280, 720):
        parse_func = parse_time_720
    elif (width, height) == (1920, 1080):
        parse_func = parse_time_1080
    else:
        print(f"[WARNING] Unknown resolution ({width}x{height}), defaulting to 720 format: {in_excel}")
        parse_func = parse_time_720

    formatted_times = []
    last_dt = None  # Tracks last valid/fixed datetime
    time_step = datetime.timedelta(seconds=0.033333)

    for idx, row in df.iterrows():
        raw_time_str = str(row["image_original_time"]).strip()
        if not raw_time_str:
            # Empty string => use last_dt + 0.033333 or leave blank
            if last_dt is not None:
                new_dt = last_dt + time_step
                ft = format_datetime(new_dt)
                formatted_times.append(ft)
                last_dt = new_dt
                print(f"[FIXED] Row {idx+1}: Empty string => previous time +0.033333 => {ft}")
                global_fix_count += 1
            else:
                formatted_times.append("")
                global_unfix_count += 1
                unfixable_details.append((video_id, idx + 1, raw_time_str))
                print(f"[WARNING] Row {idx+1}: Empty string and no previous time => left blank (unfixable)")
            continue

        # Try normal parsing
        try:
            dt = parse_func(raw_time_str)
        except Exception:
            dt = None

        if dt is None:
            # Parsing failed => try using last_dt + 0.033333 or leave blank
            if last_dt is not None:
                new_dt = last_dt + time_step
                ft = format_datetime(new_dt)
                formatted_times.append(ft)
                last_dt = new_dt
                print(f"[FIXED] Row {idx+1}: Failed to parse '{raw_time_str}', using previous time +0.033333 => {ft}")
                global_fix_count += 1
            else:
                formatted_times.append("")
                global_unfix_count += 1
                unfixable_details.append((video_id, idx + 1, raw_time_str))
                print(f"[WARNING] Row {idx+1}: Failed to parse '{raw_time_str}', no previous time => left blank (unfixable)")
            continue

        # Successfully parsed
        ft = format_datetime(dt)
        formatted_times.append(ft)
        last_dt = dt

    df["image_formatted_time"] = formatted_times

    df.to_excel(out_excel, index=False)
    print(f"[INFO] Processed and saved: {out_excel}")

def main():
    # 1) Load video resolution info
    width_height_excel = "/media/robert/4TB-SSD/video_width_height.xlsx"
    video2size = load_video_size_map(width_height_excel)

    # 2) Input/output directories
    in_root = "/media/robert/4TB-SSD/watchped_dataset"
    out_root = "/media/robert/4TB-SSD/watchped_dataset"
    os.makedirs(out_root, exist_ok=True)

    total_processed = 0

    # Iterate through all .xlsx files
    for fname in os.listdir(in_root):
        if not fname.lower().endswith(".xlsx"):
            continue

        base, _ = os.path.splitext(fname)  # e.g. "video_0001"
        in_excel_path = os.path.join(in_root, fname)
        out_excel_path = os.path.join(out_root, fname)

        # Look up resolution
        if base in video2size:
            (w, h) = video2size[base]
        else:
            print(f"[WARNING] {base} not found in video_width_height.xlsx, defaulting to (1280×720).")
            (w, h) = (1280, 720)

        # Process
        process_excel(base, in_excel_path, out_excel_path, w, h)
        total_processed += 1

    # Final summary
    print("\n[FINISHED] All processing complete. Total files processed:", total_processed)
    print(f"  Total fixed entries: {global_fix_count}")
    print(f"  Total unfixable entries: {global_unfix_count}")
    if global_unfix_count > 0:
        print("  Unfixable details:")
        for vid, row_idx, raw_str in unfixable_details:
            print(f"    - {vid}, row {row_idx}, original time = '{raw_str}'")

if __name__ == "__main__":
    import os
    main()
