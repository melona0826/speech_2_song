import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
try:
    import pickle5 as pickle
except ImportError:
    import pickle

import gc
import time
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw

TARGET_SR           = 22_050
TOP_DB              = 30
FRAME_PERIOD        = 5.0
N_FFT               = 1024
HOP_LENGTH          = 256
PITCH_THRESHOLD_MIDI= 60
GAIN_PER_SEMITONE   = 0.02


def _smart_ts(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) < 2 or target_len < 2:
        return librosa.util.fix_length(y, size=target_len)
    rate = len(y) / target_len
    D    = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    D_s  = librosa.phase_vocoder(D, rate=rate, hop_length=HOP_LENGTH)
    return librosa.istft(D_s, hop_length=HOP_LENGTH, length=target_len)

def _pitch(seg: np.ndarray, sr: int):
    seg64 = seg.astype(np.float64, copy=False)
    f0, t = pw.dio(seg64, sr, frame_period=FRAME_PERIOD)
    return f0.astype(np.float32, copy=False), t.astype(np.float32, copy=False)


def _tune_segment(args: Tuple[Dict[str, Any], int, int, np.ndarray, int]):
    seg, speech_start, sr, speech_src, hop = args
    start, end  = seg["start"], seg["end"]
    seg_len     = end - start

    raw_seg = speech_src[speech_start : speech_start + seg_len]
    seg_ts  = _smart_ts(raw_seg, seg_len).astype(np.float32, copy=False)

    f0_s, t_s = _pitch(seg_ts, sr)
    seg_ts64  = seg_ts.astype(np.float64, copy=False)
    sp_s      = pw.cheaptrick(seg_ts64, f0_s.astype(np.float64), t_s.astype(np.float64), sr)
    ap_s      = pw.d4c       (seg_ts64, f0_s.astype(np.float64), t_s.astype(np.float64), sr)
    del seg_ts64; gc.collect()

    f0_v = seg["f0_v"].astype(np.float32, copy=False)
    t_v  = seg["t_v"].astype(np.float32, copy=False)
    f0_v_interp = np.interp(t_s, t_v, f0_v, left=0, right=0)

    seg_pw = pw.synthesize(
        f0_v_interp.astype(np.float64), sp_s, ap_s, sr
    ).astype(np.float32, copy=False)

    return start, end, seg_pw

def process(cfg: Dict[str, str]):
    with open(cfg["vocal_data"], "rb") as f:
        vd: Dict[str, Any] = pickle.load(f)
    sr          = vd["sr"]
    segments    = vd["segments"]
    nonsilent   = vd["nonsilent"]
    mfcc_v      = vd["mfcc"].astype(np.float32, copy=False)

    def _load(path: str) -> np.ndarray:
        y, orig_sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        if orig_sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=TARGET_SR)
        return y

    vocals = _load(cfg["vocal_wav"])
    speech = _load(cfg["speech"])
    mr     = _load(cfg["mr"])
    sr     = TARGET_SR
    print("Load End")
    speech_ns  = librosa.effects.split(speech, top_db=TOP_DB)
    speech_all = np.concatenate([speech[s:e] for s, e in speech_ns]).astype(np.float32, copy=False)
    print("Split End")

    mfcc_s = librosa.feature.mfcc(
        y=speech_all, sr=sr, n_mfcc=13, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32, copy=False)
    print("mfcc End")
    _, wp   = librosa.sequence.dtw(X=mfcc_s, Y=mfcc_v, subseq=True)
    inv_map = {v: s for s, v in wp}
    print("dtw End")

    jobs = []
    ptr  = 0
    for seg in segments:
        start, end = seg["start"], seg["end"]
        v_frame = start // HOP_LENGTH
        s_frame = inv_map.get(v_frame, None)
        speech_start = (s_frame * HOP_LENGTH) if s_frame is not None else ptr
        ptr = speech_start + (end - start)
        jobs.append((seg, speech_start, sr, speech_all, HOP_LENGTH))

    with ThreadPoolExecutor(max_workers=4) as ex:
        tuned_raw = list(ex.map(_tune_segment, jobs))

    tuned_raw.sort(key=lambda x: x[0])

    tuned_segments: List[np.ndarray] = []
    for (start, end, seg_pw), seg in zip(tuned_raw, segments):
        rms_v = np.sqrt(np.mean(vocals[start:end] ** 2, dtype=np.float32))
        rms_s = np.sqrt(np.mean(seg_pw        ** 2, dtype=np.float32))
        if rms_s > 1e-8:
            seg_pw *= rms_v / rms_s

        midi_vals = librosa.hz_to_midi(seg["f0_v"][seg["f0_v"] > 0])
        if midi_vals.size:
            extra = max(0.0, np.median(midi_vals) - PITCH_THRESHOLD_MIDI)
            seg_pw *= 1.0 + GAIN_PER_SEMITONE * extra

        tuned_segments.append(seg_pw)

    print("segment End")

    output = []
    last   = 0
    for (ns_start, ns_end), tuned in zip(nonsilent, tuned_segments):
        output.append(np.zeros(ns_start - last, dtype=np.float32))
        output.append(tuned)
        last = ns_end
    output.append(np.zeros(len(vocals) - last, dtype=np.float32))
    final_audio = np.concatenate(output).astype(np.float32, copy=False)

    sf.write("speech_tuned.wav", final_audio, sr, subtype="PCM_16")
    print("speech_tuned.wav saved")

    max_len = max(len(final_audio), len(mr))
    vpad    = np.pad(final_audio, (0, max_len - len(final_audio)))
    mpad    = np.pad(mr,         (0, max_len - len(mr))) * (10 ** (-3 / 20))
    mixed   = vpad + mpad
    peak    = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write("final_tuned.wav", mixed.astype(np.float32, copy=False), sr, subtype="PCM_16")
    print("final_tuned.wav saved")

if __name__ == "__main__":
    t0 = time.time()
    process(
        {
            "speech":      "speech.wav",
            "mr":          "./mr/bam_yang_gang/mr.wav",
            "vocal_data":  "./vocal_data/bam_yang_gang/vocal_data.pkl",
            "vocal_wav":   "./vocals/bam_yang_gang/vocals.wav",
        }
    )
    print(f"Done in {time.time() - t0:.2f}s")
