import os
import pandas as pd

def main():
    root_dir = "/media/robert/4TB-SSD/watchped_dataset/Sensor_video_acc补充"
    gravity_value = 9.80655  # Gravitational acceleration to subtract

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # Only process directories

        acc_file = os.path.join(folder_path, "WEAR_ACC.csv")
        acg_file = os.path.join(folder_path, "WEAR_ACG.csv")

        # 1) Check if WEAR_ACC.csv is missing
        if not os.path.exists(acc_file):
            print(f"[{folder_name}] Missing WEAR_ACC.csv")

            # 2) Check if WEAR_ACG.csv exists
            if os.path.exists(acg_file):
                print(f"  -> Preparing to generate new WEAR_ACC.csv from WEAR_ACG.csv")

                try:
                    # 3) Read WEAR_ACG.csv
                    #    If the file is whitespace/tab-delimited and has no header, use:
                    df_acg = pd.read_csv(
                        acg_file,
                        delim_whitespace=True,  # If delimited by whitespace
                        header=None,            # No header in the file
                        names=['t', 'x', 'y', 'z', 'a']  # Specify 5 columns
                    )

                    # If it is actually comma-delimited CSV with no header, use:
                    # df_acg = pd.read_csv(acg_file, sep=',', header=None, names=['t','x','y','z','a'])

                    # 4) Subtract 9.80655 from the 'z' column
                    df_acg['z'] = df_acg['z'] - gravity_value

                    # 5) Save the result as WEAR_ACC.csv (5 columns: t x y z a)
                    df_acg.to_csv(acc_file, sep=' ', index=False, header=False)
                    print(f"  -> New WEAR_ACC.csv generated at {acc_file}")

                except Exception as e:
                    print(f"  -> Error reading/generating file: {e}")
            else:
                print(f"  -> WEAR_ACG.csv also missing, cannot generate WEAR_ACC.csv")
        else:
            # Already exists, no need to process
            print(f"[{folder_name}] WEAR_ACC.csv already exists, no need to supplement")

if __name__ == "__main__":
    main()
