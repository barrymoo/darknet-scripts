#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

from scipy import signal
from librosa import load

from sklearn.preprocessing import MinMaxScaler
from PIL import Image


def chunk_preprocess(chunk):
    for label in chunk:
        write_image(label)
    return


def decibel_filter(spectrogram, db_cutoff=-100.0):
    remove_zeros = np.copy(spectrogram)
    remove_zeros[remove_zeros == 0.0] = np.nan
    inDb = 10.0 * np.log10(remove_zeros)
    inDb[inDb <= db_cutoff] = db_cutoff
    return np.nan_to_num(10.0 ** (inDb / 10.0))


def write_image(label):
    # Generate spectrogram
    samples, sample_rate = load(
        f"/media/data/new-towhee-scapes/{label}",
        mono = False,
        sr = 22050,
        res_type = "kaiser_fast",
    )
    freq, time, spec = signal.spectrogram(
        samples,
        sample_rate,
        window = "hann",
        nperseg = 512,
        noverlap = 384,
        nfft = 512,
        scaling = "spectrum",
    )

    # Anything lower than -100dB is removed
    spec = decibel_filter(spec)

    # Take the log of the spectrogram intensities
    spec = np.log10(spec)

    # Simple z-score normalization
    spec_mean = np.mean(spec)
    spec_std = np.std(spec)
    spec = (spec - spec_mean) / spec_std

    # Get the filename
    p = Path(f"{label}")

    # Save the figure
    scaler = MinMaxScaler(feature_range=(0, 255))
    spec = scaler.fit_transform(spec)
    image = Image.fromarray(np.flip(spec, axis=0))
    image = image.convert("RGB")
    image = image.resize((832, 256))
    image.save(f"/media/data/towhee/JPEGImages/{p.stem}.jpg")


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
