import pickle
import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
import gc
import time

TOP_DB = 30
FRAME_PERIOD = 5.0
PITCH_THRESHOLD_MIDI = 60
GAIN_PER_SEMITONE = 0.02


def smart_time_stretch(y: np.ndarray, target_len: int) -> np.ndarray:
    orig_len = len(y)
    if orig_len < 2 or target_len < 2:
        return librosa.util.fix_length(y, size=target_len)

    rate = orig_len / target_len
    power = int(np.floor(np.log2(orig_len)))
    n_fft = min(2048, 2 ** power)
    hop_len = n_fft // 4

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    D_stretch = librosa.phase_vocoder(D, rate=rate, hop_length=hop_len)
    return librosa.istft(D_stretch, hop_length=hop_len, length=target_len)


def process(speech_path: str, mr_path: str, vocal_data_path: str, vocal_wav_path: str):
    with open(vocal_data_path, "rb") as f:
        vd = pickle.load(f)

    sr = vd["sr"]
    vocal_nonsilent = vd["vocal_nonsilent"]
    segments = vd["segments"]
    mfcc_v = vd["mfcc_v"]

    vocals, _ = librosa.load(vocal_wav_path, sr=sr, dtype=np.float32, mono=True)
    speech, _ = librosa.load(speech_path, sr=sr, dtype=np.float32, mono=True)
    mr, _ = librosa.load(mr_path, sr=sr, dtype=np.float32, mono=True)

    speech_ns = librosa.effects.split(speech, top_db=TOP_DB)
    speech_ns_audio = np.concatenate([speech[s:e] for s, e in speech_ns]).astype(
        np.float32, copy=False
    )

    mfcc_s = librosa.feature.mfcc(
        y=speech_ns_audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512
    ).astype(np.float32, copy=False)
    _, wp = librosa.sequence.dtw(X=mfcc_s, Y=mfcc_v, subseq=True)
    inv_map = {v: s for s, v in wp}

    tuned_segments = []
    ptr = 0
    hop = 512

    for seg_info in segments:
        start, end = seg_info["start"], seg_info["end"]
        seg_len = end - start

        vframe = int(start / hop)
        sframe = inv_map.get(vframe)
        speech_start = (sframe * hop) if (sframe is not None) else ptr
        ptr = speech_start + seg_len

        raw_seg = speech_ns_audio[speech_start:ptr]
        seg_ts = smart_time_stretch(raw_seg, seg_len).astype(np.float32, copy=False)

        seg_ts64 = seg_ts.astype(np.float64, copy=False)  # pyworld requirement
        f0_s, t_s = pw.harvest(seg_ts64, sr, frame_period=FRAME_PERIOD)
        sp_s = pw.cheaptrick(seg_ts64, f0_s, t_s, sr)
        ap_s = pw.d4c(seg_ts64, f0_s, t_s, sr)

        del seg_ts64
        gc.collect()

        f0_v = seg_info["f0_v"].astype(np.float64, copy=False)
        t_v = seg_info["t_v"].astype(np.float64, copy=False)
        f0_v_interp = np.interp(t_s, t_v, f0_v, left=0, right=0)

        seg_pw = pw.synthesize(f0_v_interp, sp_s, ap_s, sr).astype(np.float32, copy=False)

        vocal_seg = vocals[start:end]
        rms_vocal = np.sqrt(np.mean(vocal_seg ** 2, dtype=np.float32))
        rms_synth = np.sqrt(np.mean(seg_pw ** 2, dtype=np.float32))
        if rms_synth > 1e-8:
            seg_pw *= rms_vocal / rms_synth

        midi_v = librosa.hz_to_midi(f0_v[f0_v > 0])
        if midi_v.size:
            median_midi = np.median(midi_v)
            extra = max(0.0, median_midi - PITCH_THRESHOLD_MIDI)
            seg_pw *= 1.0 + GAIN_PER_SEMITONE * extra

        tuned_segments.append(seg_pw)
        del f0_s, t_s, sp_s, ap_s, f0_v, t_v, seg_pw
        gc.collect()

    output = []
    last = 0
    for (start, end), tuned in zip(vocal_nonsilent, tuned_segments):
        output.append(np.zeros(start - last, dtype=np.float32))
        output.append(tuned)
        last = end
    output.append(np.zeros(len(vocals) - last, dtype=np.float32))
    final_audio = np.concatenate(output).astype(np.float32, copy=False)

    sf.write("speech_tuned.wav", final_audio, sr, subtype="PCM_16")
    print("speech_tuned.wav saved")

    max_len = max(len(final_audio), len(mr))
    vpad = np.pad(final_audio, (0, max_len - len(final_audio)))
    mpad = np.pad(mr, (0, max_len - len(mr)))

    db2amp = lambda db: 10.0 ** (db / 20.0)
    vpad *= db2amp(0.0)
    mpad *= db2amp(-3.0)

    mixed = vpad + mpad
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write("final_tuned.wav", mixed.astype(np.float32, copy=False), sr, subtype="PCM_16")
    print("final_tuned.wav saved")


if __name__ == "__main__":
    start_time = time.time()
    process(
        speech_path="speech.wav",
        mr_path="./mr/bam_yang_gang/mr.wav",
        vocal_data_path="./vocal_data/bam_yang_gang/vocal_data.pkl",
        vocal_wav_path="./vocals/bam_yang_gang/vocals.wav",
    )
    print(f"Conversion Time : {time.time() - start_time:.2f}s")
