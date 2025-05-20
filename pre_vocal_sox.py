import pickle
import numpy as np
import librosa
import pyworld as pw
import pathlib
import os
import gc

TOP_DB = 30
FRAME_PERIOD = 5.0


def preprocess_vocal(vocal_path: str, out_path: str):

    vocals, sr = librosa.load(vocal_path, sr=None, dtype=np.float32, mono=True)

    vocal_nonsilent = librosa.effects.split(vocals, top_db=TOP_DB)

    segments = []
    for start, end in vocal_nonsilent:
        seg = vocals[start:end]

        seg64 = seg.astype(np.float64, copy=False)
        f0_v, t_v = pw.harvest(seg64, sr, frame_period=FRAME_PERIOD)
        segments.append(
            {
                "start": int(start),
                "end": int(end),
                "f0_v": f0_v.astype(np.float32, copy=False),
                "t_v": t_v.astype(np.float32, copy=False),
            }
        )
        del seg64
        gc.collect()

    mfcc_v = librosa.feature.mfcc(
        y=vocals,
        sr=sr,
        n_mfcc=13,
        n_fft=2048,
        hop_length=512,
    ).astype(np.float32, copy=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "sr": sr,
                "vocal_nonsilent": vocal_nonsilent,
                "segments": segments,
                "mfcc_v": mfcc_v,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Vocal data saved to {out_path}")


if __name__ == "__main__":
    base_dir = pathlib.Path(__file__).parent.resolve()
    songs_dir = base_dir / "songs"
    out_root = base_dir / "vocal_data"

    for song_dir in songs_dir.iterdir():
        if song_dir.is_dir():
            vocal_path = song_dir / "vocals.wav"
            if vocal_path.exists():
                out_path = out_root / song_dir.name / "vocal_data.pkl"
                try:
                    preprocess_vocal(str(vocal_path), str(out_path))
                except Exception as e:
                    print(f"Failed for {vocal_path}: {e}")
            else:
                print(f"(skip) {vocal_path} not found")
