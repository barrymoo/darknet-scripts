#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from scipy import signal
from librosa import load

from sklearn.preprocessing import MinMaxScaler
from PIL import Image

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import json


def decibel_filter(spectrogram, db_cutoff=-100.0):
    remove_zeros = np.copy(spectrogram)
    remove_zeros[remove_zeros == 0.0] = np.nan
    inDb = 10.0 * np.log10(remove_zeros)
    inDb[inDb <= db_cutoff] = db_cutoff
    return np.nan_to_num(10.0 ** (inDb / 10.0))


def chunk_preprocess(chunk):
    results = [None] * chunk.shape[0]
    for idx, (_, row) in enumerate(chunk.iterrows()):
        results[idx] = (row["Index"], preprocess(row["Filename"]))
    return results


def preprocess(filename):
    # The path for p.stem
    p = Path(filename)

    # Generate frequencies and times
    samples, sample_rate = load(
        f"{base}/{p.parent}/{p.stem}.wav", mono=False, sr=22050, res_type="kaiser_fast"
    )
    freq, time, spec = signal.spectrogram(
        samples,
        sample_rate,
        window="hann",
        nperseg=512,
        noverlap=384,
        nfft=512,
        scaling="spectrum",
    )

    # Filters
    spec = decibel_filter(spec)
    spec = np.log10(spec)
    spec_mean = np.mean(spec)
    spec_std = np.std(spec)
    spec = (spec - spec_mean) / spec_std

    # Scale the image
    scaler = MinMaxScaler(feature_range=(0, 255))
    spec = scaler.fit_transform(spec)

    # Split into 5 second overlapping sections
    # -> w/ 1 second overlap
    duration = np.max(time) - np.min(time)
    length_px = int(spec.shape[1])
    px_per_sec = int(length_px / duration)
    overlap_px = int(4 * px_per_sec)
    window_px = int(5 * px_per_sec)
    splits = []
    for begin in range(0, length_px - overlap_px, window_px - overlap_px):
         if begin + window_px > length_px:
             end = length_px - 1
             begin = length_px - window_px - 1
         else:
             end = begin + window_px
         splits.append((begin, end))

    # Read the corresponding CSV
    inner = pd.read_csv(f"{base}/{p.parent}/{p.stem}.csv")

    results = [None] * len(splits)
    for suffix, (begin, end) in enumerate(splits):
        img_f = f"{base}/overlaps/{p.parent}-{p.stem}-{suffix}.jpg"

        # Write the image
        img = Image.fromarray(np.flip(spec[:, begin:end], axis=0))
        img = img.convert("RGB")
        # Resize to InceptionNet Default size
        img = img.resize((299, 299))
        img.save(img_f)

        labels = set()
        for idx, row in inner.iterrows():
            # Generate labels
            x_mins = json.loads(row["Start_times"])
            x_maxs = json.loads(row["End_times"])

            for x_min, x_max in zip(x_mins, x_maxs):
                if x_min > time[begin] and x_min < time[end]:
                    labels.add(row["Bird"])
                elif x_max > time[begin] and x_max < time[end]:
                    labels.add(row["Bird"])

        results[suffix] = (img_f, list(labels))
    return results

base = "/media/powdermill/new_scapes"

names = pd.read_csv("pnre.names", header=None)

df = pd.read_csv(f"{base}/wav.files.csv")
df["Index"] = df.index.values

stride = 57

results = pd.DataFrame(index=range(stride * df.shape[0]), columns=["X", "y"], dtype=str)

nprocs = cpu_count()
chunks = np.array_split(df[["Filename", "Index"]], nprocs)

executor = ProcessPoolExecutor(nprocs)
futs = [executor.submit(chunk_preprocess, chunk) for chunk in chunks]
for fut in as_completed(futs):
    res = fut.result()
    for outer_idx, ll in res:
        for inner_idx, (X, y) in enumerate(ll):
            results.loc[outer_idx * stride + inner_idx, "X"] = X
            results.loc[outer_idx * stride + inner_idx, "y"] = y

results.to_csv("all_files.csv", index=None)
