import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
try:
    import pickle5 as pickle
except ImportError:
    import pickle

import gc
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import pyworld as pw

TARGET_SR          = 22_050
TOP_DB             = 30
FRAME_PERIOD       = 5.0       # ms
N_FFT              = 1024
HOP_LENGTH         = 256

def _extract_pitch(seg: np.ndarray, sr: int):
    seg64 = seg.astype(np.float64, copy=False)
    f0, t = pw.dio(seg64, sr, frame_period=FRAME_PERIOD)
    return f0.astype(np.float32, copy=False), t.astype(np.float32, copy=False)

def preprocess_vocal(vocal_path: pathlib.Path, out_path: pathlib.Path):
    y, sr = librosa.load(vocal_path, sr=None, mono=True, dtype=np.float32)
    if sr != TARGET_SR:
        y  = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    nonsilent = librosa.effects.split(y, top_db=TOP_DB)

    with ThreadPoolExecutor(max_workers=4) as ex:
        pitch = list(ex.map(lambda span: _extract_pitch(y[span[0]:span[1]], sr), nonsilent))

    segments = [
        {"start": int(s), "end": int(e), "f0_v": f0, "t_v": t}
        for (s, e), (f0, t) in zip(nonsilent, pitch)
    ]

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32, copy=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    proto = 4 if sys.version_info < (3, 8) else pickle.HIGHEST_PROTOCOL
    with open(out_path, "wb") as f:
        pickle.dump(
            {"sr": sr, "nonsilent": nonsilent, "segments": segments, "mfcc": mfcc},
            f,
            protocol=proto,
        )
    print("vocal_data.pkl saved : ", out_path)

if __name__ == "__main__":
    base = pathlib.Path(__file__).parent.resolve()
    for song in (base / "songs").iterdir():
        wav = song / "vocals.wav"
        if wav.exists():
            preprocess_vocal(
                wav,
                base / "vocal_data" / song.name / "vocal_data.pkl",
            )
