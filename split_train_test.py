#!/usr/bin/env python3
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# Read the CSV file, I only need the "Filename" column
df = pd.read_csv("towhee.csv")

# Generate the number of boxes
def return_length(row):
    l = json.loads(row["Start_times"])
    return len(l)
df["NumBoxes"] = df.apply(return_length, axis=1)

# Remove cases where there are no boxes
df = df[df["NumBoxes"] != 0]

# Fix the Filename column
def fix_path(filename):
    p = Path(filename)
    return f"/media/data/towhee/JPEGImages/{p.stem}.jpg"
df["Filename"] = df["Filename"].apply(fix_path)

train, test = train_test_split(df["Filename"], test_size=0.33)

train.to_csv("train.txt", header=False, index=False)
test.to_csv("test.txt", header=False, index=False)
