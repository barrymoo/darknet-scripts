#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from PIL import Image


def chunk_preprocess(chunk):
    for label in chunk:
        check_image(label)
    return


def check_image(label):
    p = Path(label)
    image = Image.open(f"/media/data/towhee/JPEGImages/{p.stem}.jpg")
    assert(image.size == (832, 256))


# Read the CSV file, I only need the "Filename" column
df = pd.read_csv("towhee.csv")
filenames = df["Filename"]

# Split into chunks
nprocs = cpu_count()
chunks = np.array_split(filenames, nprocs)

# Run in parallel
executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(chunk_preprocess, chunk) for chunk in chunks]
for fut in as_completed(futs):
    _ = fut.result()
