#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
import json

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from scipy import signal
from librosa import load

def chunk_preprocess(chunk):
    for _, row in chunk.iterrows():
        write_label(row)
    return


def write_label(row):
    # Extract label
    label = row["Filename"]
    p = Path(label)

    # Generate frequencies and times
    samples, sample_rate = load(
        f"/media/data/new-towhee-scapes/{label}",
        mono = False,
        sr = 22050,
        res_type = "kaiser_fast",
    )
    freq, time, _ = signal.spectrogram(
        samples,
        sample_rate,
        window = "hann",
        nperseg = 512,
        noverlap = 384,
        nfft = 512,
        scaling = "spectrum",
    )

    # Generate information
    x_mins = json.loads(row["Start_times"])
    x_maxs = json.loads(row["End_times"])
    y_mins = json.loads(row["Freq_lows"])
    y_maxs = json.loads(row["Freq_highs"])

    time_min = np.min(time)
    time_max = np.max(time)
    freq_min = np.min(freq)
    freq_max = np.max(freq)
    y_length = freq_max - freq_min
    x_length = time_max - time_min

    with open(f"/media/data/towhee/labels/{p.stem}.txt", "w") as f:
        for x_min, x_max, y_min, y_max in zip(x_mins, x_maxs, y_mins, y_maxs):
            if x_min < time_min:
                x_min = time_min
            if x_max > time_max:
                x_max = time_max
            x = x_min / x_length
            y = y_min / y_length
            width = (x_max - x_min) / x_length
            height = (y_max - y_min) / y_length
            if x + width > 1.0:
                width = 1.0 - x
            if y + height > 1.0:
                height = 1.0 - y
            assert(x + width <= 1.0 and y + height <= 1.0)
            f.write(f"0 {x + 0.5 * width} {y + 0.5 * height} {width} {height}\n")

# Read the CSV file, I only need the "Filename" column
df = pd.read_csv("/media/data/new-towhee-scapes/towhees.csv")

# Split into chunks
nprocs = cpu_count()
chunks = np.array_split(df, nprocs)

# Run in parallel
executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(chunk_preprocess, chunk) for chunk in chunks]
for fut in as_completed(futs):
    _ = fut.result()
