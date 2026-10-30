"""Stage 2 – windowed features, each signal at its native rate.

EDA/TEMP from the 4 Hz file, ACC from the 32 Hz file, BVP from the 64 Hz file (the data article
advises against BVP below 64 Hz for beat-based analysis). Windows never cross a phase segment.
Usage: python s02_features.py [window_s ...]   (default 30; 50 % overlap)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RES, SUBJECTS, load_demographics, load_processed, segments  # noqa: E402


def slope(v, fs):
    if len(v) < 2:
        return np.nan
    return np.polyfit(np.arange(len(v)) / fs, v, 1)[0]


def eda_features(e, fs=4):
    tonic = median_filter(e, size=min(len(e), 8 * fs), mode="nearest")  # 8 s running median = tonic level
    phasic = e - tonic
    peaks, props = signal.find_peaks(phasic, prominence=0.01)
    dur_min = len(e) / fs / 60
    return {
        "eda_mean": e.mean(), "eda_std": e.std(), "eda_range": np.ptp(e), "eda_slope": slope(e, fs),
        "eda_tonic_mean": tonic.mean(), "eda_phasic_std": phasic.std(),
        "eda_scr_rate": len(peaks) / dur_min,
        "eda_scr_amp": props["prominences"].mean() if len(peaks) else 0.0,
        "eda_absdiff": np.abs(np.diff(e)).mean() * fs,
    }


def temp_features(t, fs=4):
    return {"temp_mean": t.mean(), "temp_std": t.std(), "temp_slope": slope(t, fs)}


def acc_features(xyz, fs=32):
    g = xyz / 64.0  # E4 units are 1/64 g
    mag = np.sqrt((g ** 2).sum(axis=1))
    f, p = signal.welch(mag - mag.mean(), fs=fs, nperseg=min(len(mag), 8 * fs))
    band = (f >= 0.1) & (f <= 10)
    return {
        "acc_mag_mean": mag.mean(), "acc_mag_std": mag.std(),
        "acc_enmo": np.clip(mag - 1, 0, None).mean(),
        "acc_x_std": g[:, 0].std(), "acc_y_std": g[:, 1].std(), "acc_z_std": g[:, 2].std(),
        "acc_jerk": np.abs(np.diff(mag)).mean() * fs,
        "acc_dom_freq": f[band][np.argmax(p[band])] if band.any() else np.nan,
    }


_BP = signal.butter(3, [0.5, 4.0], btype="band", fs=64, output="sos")


def bvp_features(b, fs=64):
    """Pulse rate / HRV from BVP peaks with simple physiological quality gates."""
    out = dict.fromkeys(["bvp_amp_std", "hr_mean", "hr_std", "ibi_sdnn", "ibi_rmssd", "ibi_pnn50", "bvp_quality"], np.nan)
    if len(b) < 10 * fs:
        return out
    x = signal.sosfiltfilt(_BP, b)
    out["bvp_amp_std"] = x.std()
    peaks, _ = signal.find_peaks(x, distance=int(0.3 * fs), prominence=0.3 * x.std())
    ibi = np.diff(peaks) / fs
    ok = (ibi > 0.3) & (ibi < 1.5)  # 40–200 bpm
    if ok.sum() >= 2:  # reject IBIs that differ > 30 % from the median (ectopic / artefact)
        med = np.median(ibi[ok])
        ok &= np.abs(ibi - med) < 0.3 * med
    out["bvp_quality"] = ok.mean() if len(ibi) else 0.0
    good = ibi[ok]
    if len(good) >= 5:
        hr = 60 / good
        succ = np.abs(np.diff(ibi))[ok[1:] & ok[:-1]]  # only pairs of adjacent valid beats
        out.update(hr_mean=hr.mean(), hr_std=hr.std(), ibi_sdnn=good.std() * 1000,
                   ibi_rmssd=np.sqrt(np.mean(succ ** 2)) * 1000 if len(succ) else np.nan,
                   ibi_pnn50=(succ > 0.05).mean() if len(succ) else np.nan)
    return out


def subject_windows(sid, win_s, step_s):
    d4, d32, d64 = (load_processed(sid, hz) for hz in (4, 32, 64))
    s4, s32, s64 = segments(d4, 4), segments(d32, 32), segments(d64, 64)
    assert len(s4) == len(s32) == len(s64), f"{sid}: segment mismatch"
    rows = []
    for k in range(len(s4)):
        r4, r32, r64 = s4.iloc[k], s32.iloc[k], s64.iloc[k]
        dur = (r4.stop - r4.start) / 4
        # segments shorter than one window contribute a single (shorter) window if >= 10 s
        starts = np.arange(0, max(dur - win_s, 0) + 1e-9, step_s) if dur >= win_s else ([0.0] if dur >= 10 else [])
        for t0 in starts:
            t1 = min(t0 + win_s, dur)
            sl = lambda r, fs: slice(int(r.start + t0 * fs), int(r.start + t1 * fs))  # noqa: E731
            e = d4.eda.to_numpy()[sl(r4, 4)]
            t = d4.temperature.to_numpy()[sl(r4, 4)]
            a = d32[["x", "y", "z"]].to_numpy()[sl(r32, 32)]
            b = d64.bvp.to_numpy()[sl(r64, 64)]
            f = {"subject": sid, "seg": k, "phase": r4.phase, "pain": int(r4.pain), "t0": t0, "win_s": t1 - t0}
            f.update(eda_features(e)); f.update(temp_features(t)); f.update(acc_features(a)); f.update(bvp_features(b))
            rows.append(f)
    return rows


def build(win_s):
    step_s = win_s / 2
    rows = [r for sid in SUBJECTS for r in subject_windows(sid, win_s, step_s)]
    df = pd.DataFrame(rows)
    demo = load_demographics()[["subject", "sex", "age", "diag_group"]]
    df = df.merge(demo, on="subject", how="left")
    df["is_exercise"] = (df.phase == "exercise").astype(int)
    path = os.path.join(RES, f"features_w{int(win_s)}.csv")
    df.to_csv(path, index=False)
    print(f"window {win_s}s: {len(df)} windows, {df.subject.nunique()} subjects, "
          f"{df.groupby(['subject', 'seg']).ngroups} segments -> {path}")
    print("  HR available in %.0f%% of windows; median BVP quality %.2f" % (
        100 * df.hr_mean.notna().mean(), df.bvp_quality.median()))
    return df


if __name__ == "__main__":
    for w in [float(a) for a in sys.argv[1:]] or [30.0]:
        build(w)
