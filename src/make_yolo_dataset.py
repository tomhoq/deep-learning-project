import pandas as pd
import os
import sys
from models.yolo.dataframe import get_dataframe


# Function to convert to YOLO format
def convert_to_yolo(df):
    # Class ID (assuming single class, e.g., "0")
    class_id = 0

    yolo_data = {}
    for image_id, group in df.groupby("ImageId"):
        yolo_lines = []
        for _, row in group.iterrows():
            bbox = row["Bbox"]
            x_center, y_center, width, height = bbox
            yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

        yolo_data[image_id] = "\n".join(yolo_lines)
        print("*", end="")

    print("+")

    return yolo_data


def df2txt(mode):
    df = get_dataframe()[mode];

    # Save to .txt files
    for image_id, annotation in convert_to_yolo(df).items():
        txt_filename = image_id.replace(".jpg", ".txt")
        path = os.path.join(os.getenv('BLACKHOLE'), 'yolo', mode, txt_filename)
        with open(path, "w") as f:
            f.write(annotation)

        print(".", end="")


    print(f"\n[+] Mode {mode} done\n\n")


df2txt('train');
df2txt('validation');






