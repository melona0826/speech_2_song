import pickle

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw

import time

TOP_DB              = 30
FRAME_PERIOD        = 5.0
PITCH_THRESHOLD_MIDI = 60
GAIN_PER_SEMITONE    = 0.02

def smart_time_stretch(y, target_len):
    orig_len = len(y)
    if orig_len < 2 or target_len < 2:
        return librosa.util.fix_length(y, size=target_len)
    rate = orig_len / target_len
    power = int(np.floor(np.log2(orig_len)))
    n_fft = min(2048, 2**power)
    hop_len = n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_len)
    D_stretch = librosa.phase_vocoder(D, rate=rate, hop_length=hop_len)
    return librosa.istft(D_stretch, hop_length=hop_len, length=target_len)

def process(speech_path, mr_path, vocal_data_path, vocal_wav_path):
    with open(vocal_data_path, 'rb') as f:
        vd = pickle.load(f)
    sr               = vd['sr']
    vocal_nonsilent  = vd['vocal_nonsilent']
    segments         = vd['segments']
    mfcc_v           = vd['mfcc_v']

    vocals, _ = librosa.load(vocal_wav_path, sr=sr)
    vocals = vocals.astype(np.float64)

    speech, _ = librosa.load(speech_path, sr=sr)
    speech = speech.astype(np.float64)
    mr, _   = librosa.load(mr_path,    sr=sr)
    mr = mr.astype(np.float64)

    speech_ns = librosa.effects.split(speech, top_db=TOP_DB)
    speech_ns_audio = np.concatenate([speech[s:e] for s,e in speech_ns]).astype(np.float64)

    mfcc_s = librosa.feature.mfcc(
        y=speech_ns_audio, sr=sr, n_mfcc=13,
        n_fft=2048, hop_length=512
    ).astype(np.float32)
    _, wp = librosa.sequence.dtw(X=mfcc_s, Y=mfcc_v, subseq=True)
    inv_map = {v: s for s, v in wp}

    tuned_segments = []
    ptr = 0
    hop = 512

    for seg_info in segments:
        start, end = seg_info['start'], seg_info['end']
        seg_len = end - start

        vframe = int(start / hop)
        sframe = inv_map.get(vframe)
        speech_start = (sframe * hop) if (sframe is not None) else ptr
        ptr = speech_start + seg_len

        raw_seg = speech_ns_audio[speech_start:ptr]
        seg_ts  = smart_time_stretch(raw_seg, seg_len).astype(np.float64)

        f0_s, t_s = pw.harvest(seg_ts, sr, frame_period=FRAME_PERIOD)
        sp_s      = pw.cheaptrick(seg_ts, f0_s, t_s, sr)
        ap_s      = pw.d4c(seg_ts,       f0_s, t_s, sr)

        f0_v      = seg_info['f0_v']
        t_v       = seg_info['t_v']
        f0_v_interp = np.interp(t_s, t_v, f0_v, left=0, right=0)

        seg_pw = pw.synthesize(f0_v_interp, sp_s, ap_s, sr)

        vocal_seg = vocals[start:end]
        rms_vocal = np.sqrt(np.mean(vocal_seg**2))
        rms_synth = np.sqrt(np.mean(seg_pw**2))
        if rms_synth > 1e-8:
            seg_pw *= (rms_vocal / rms_synth)

        midi_v = librosa.hz_to_midi(f0_v[f0_v>0])
        if midi_v.size > 0:
            median_midi = np.median(midi_v)
            extra = max(0.0, median_midi - PITCH_THRESHOLD_MIDI)
            seg_pw *= (1.0 + GAIN_PER_SEMITONE * extra)

        tuned_segments.append(seg_pw)

    output = []
    last = 0
    for (start, end), tuned in zip(vocal_nonsilent, tuned_segments):
        output.append(np.zeros(start - last, dtype=np.float64))
        output.append(tuned)
        last = end
    output.append(np.zeros(len(vocals) - last, dtype=np.float64))
    final_audio = np.concatenate(output)

    sf.write('speech_tuned.wav', final_audio, sr)
    print("✔ speech_tuned.wav 생성")

    max_len = max(len(final_audio), len(mr))
    vpad = np.pad(final_audio, (0, max_len - len(final_audio)))
    mpad = np.pad(mr,          (0, max_len - len(mr)))

    db2amp = lambda db: 10.0**(db/20.0)
    vpad *= db2amp(0)
    mpad *= db2amp(-3)

    mixed = vpad + mpad
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak

    sf.write('final_tuned.wav', mixed, sr)
    print("final_tuned.wav 생성")

if __name__ == '__main__':
    start_time = time.time()
    process(
      speech_path='speech.wav',
      mr_path='./mr/see_you_again/mr.wav',
      vocal_data_path='./vocal_data/see_you_again/vocal_data.pkl',
      vocal_wav_path='./vocals/see_you_again/vocals.wav',
    )
    print(f"Conversion Time : {time.time() - start_time}")
