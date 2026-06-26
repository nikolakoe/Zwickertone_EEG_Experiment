import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_welch
from scipy.stats import wilcoxon, friedmanchisquare

# =====================================================================
# CONFIG
# =====================================================================
CONFIG = {
    "data_path": "...",
    "noise_file_tpl": "{block}_{subj}_{cond}_raw.fif",
    "pause_file_tpl": "{block}_{subj}_{cond}_pause_raw.fif",
    "noise_blocks": [1, 2, 3],
    "conditions": ["white", "notch"],   # index 0 = white, 1 = notch
    "sfreq": 100,
    "outdir": "...",

    # analysis toggles
    "RUN_SUSTAINED": True,       # (1)
    "RUN_OFFSET_DECAY": True,    # (2)
    "RUN_ONSET_ADAPT": True,     # (3)
    "RUN_BAND_AUDITORY": True,   # (5)

    # parameters
    "SUSTAINED_SKIP_S": 2.0,     # drop onset transient from 40 s blocks
    "PAUSE_OFFSET_LAG_S": 0.0,   # seconds before true offset inside pause files
    "DECAY_BIN_S": 0.250,        # time-bin width for the pause decay trajectory
    "DECAY_TOTAL_S": 10.0,       # total pause length to analyse
    "ONSET_WIN_S": (0.0, 1.0),   # window after block onset for the onset response
    "fdr_alpha": 0.05,
}

FREQ_BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 45),
}

# _s = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]  #tinnitus
_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 25, 26, 27, 29, 35, 38, 18, 19, 20, 23, 24]  #healthy

SUBJECTS = [f"Prob{i}" for i in _s]

os.makedirs(CONFIG["outdir"], exist_ok=True)


# =====================================================================
# Helpers
# =====================================================================
def load_raw(subj, block, cond, tpl):
    fp = os.path.join(CONFIG["data_path"],
                      tpl.format(block=block, subj=subj, cond=cond))
    if not os.path.exists(fp):
        return None
    return mne.io.read_raw_fif(fp, verbose=False)


def band_power(data, sfreq, bands, fmin=1, fmax=45, n_fft=None):
    """Mean log10 PSD per band, averaged over channels.
    data: (n_channels, n_samples) -> dict band -> scalar (mean over channels)."""
    if n_fft is None:
        n_fft = min(data.shape[1], 256)
    if data.shape[1] < 8:
        return {b: np.nan for b in bands}
    psd, freqs = psd_array_welch(data*10**(6), sfreq=sfreq, fmin=fmin, fmax=fmax,
                                 n_fft=n_fft, verbose=False)
    psd = np.log10(psd + 1e-10)
    out = {}
    for b, (lo, hi) in bands.items():
        idx = np.logical_and(freqs >= lo, freqs < hi)
        out[b] = float(psd[:, idx].mean()) if idx.any() else np.nan
    return out


def band_power_per_channel(data, sfreq, bands, fmin=1, fmax=45, n_fft=None):
    """Like band_power but returns per-channel arrays: dict band -> (n_ch,)."""
    if n_fft is None:
        n_fft = min(data.shape[1], 256)
    psd, freqs = psd_array_welch(data*10**(6), sfreq=sfreq, fmin=fmin, fmax=fmax,
                                 n_fft=n_fft, verbose=False)
    psd = np.log10(psd + 1e-10)
    out = {}
    for b, (lo, hi) in bands.items():
        idx = np.logical_and(freqs >= lo, freqs < hi)
        out[b] = psd[:, idx].mean(axis=1) if idx.any() else np.full(psd.shape[0], np.nan)
    return out


def fdr_bh(pvals, alpha):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresh
    reject = np.zeros(n, bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[:kmax + 1]] = True
    return reject


# =====================================================================
# (1) Sustained-stimulation power: notch vs white
# =====================================================================
def run_sustained():
    print("\n=== (1) Sustained-stimulation power: notch vs white ===")
    sfreq = CONFIG["sfreq"]
    skip = int(CONFIG["SUSTAINED_SKIP_S"] * sfreq)
    rows = []
    for subj in SUBJECTS:
        per_cond = {}
        ok = True
        for cond in CONFIG["conditions"]:
            band_vals = {b: [] for b in FREQ_BANDS}
            for block in CONFIG["noise_blocks"]:
                raw = load_raw(subj, block, cond, CONFIG["noise_file_tpl"])
                if raw is None:
                    ok = False; break
                data = raw.get_data()[:, skip:]   # drop onset transient
                bp = band_power(data, sfreq, FREQ_BANDS)
                for b in FREQ_BANDS:
                    band_vals[b].append(bp[b])
            if not ok:
                break
            per_cond[cond] = {b: np.nanmean(band_vals[b]) for b in FREQ_BANDS}
        if not ok:
            continue
        row = {"Subject": subj}
        for b in FREQ_BANDS:
            row[f"white_{b}"] = per_cond["white"][b]
            row[f"notch_{b}"] = per_cond["notch"][b]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CONFIG["outdir"], "sustained_power.csv"), index=False)
    if df.empty:
        print("  [skip] no subjects loaded"); return

    # paired Wilcoxon per band
    stats = []
    for b in FREQ_BANDS:
        w = df[f"white_{b}"].values
        n = df[f"notch_{b}"].values
        try:
            stat, p = wilcoxon(n, w)
        except ValueError:
            stat, p = np.nan, 1.0
        stats.append({"band": b, "white_mean": np.nanmean(w),
                      "notch_mean": np.nanmean(n),
                      "diff(notch-white)": np.nanmean(n - w), "p": p})
    sdf = pd.DataFrame(stats)
    sdf["p_fdr_reject"] = fdr_bh(sdf["p"].values, CONFIG["fdr_alpha"])
    print("fdr_sustained", fdr_bh(sdf["p"].values, CONFIG["fdr_alpha"]))
    sdf.to_csv(os.path.join(CONFIG["outdir"], "sustained_power_stats.csv"), index=False)
    print(sdf.round(4).to_string(index=False))

    # plot: mean +/- SEM per band, both conditions
    bands = list(FREQ_BANDS)
    x = np.arange(len(bands)); wbar = 0.38
    wm = [df[f"white_{b}"].mean() for b in bands]
    nm = [df[f"notch_{b}"].mean() for b in bands]
    we = [df[f"white_{b}"].sem() for b in bands]
    ne = [df[f"notch_{b}"].sem() for b in bands]
    plt.figure(figsize=(4, 5))
    plt.bar(x - wbar/2, wm, wbar, yerr=we, capsize=4, label="white", color="#CC0000")
    plt.bar(x + wbar/2, nm, wbar, yerr=ne, capsize=4, label="notch", color="#0066CC")

    for i, b in enumerate(bands):
        if sdf.loc[sdf.band == b, "p_fdr_reject"].values[0]:
            plt.text(i, 0.04, "*", ha="center", fontsize=14)
    plt.xticks(x, ["δ", "θ", "α", "β", "γ"], fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim(-1.4, 0.15)
    plt.ylabel("log PSD (mean over channels)", fontsize=14)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["outdir"], "sustained_power.pdf"), dpi=150)
    plt.close()
    print("  wrote sustained_power.png")


# =====================================================================
# (2) Offset decay trajectory across the 10 s pause
# =====================================================================
def run_offset_decay():
    print("\n=== (2) Offset decay trajectory in the pauses ===")
    sfreq = CONFIG["sfreq"]
    lag = int(CONFIG["PAUSE_OFFSET_LAG_S"] * sfreq)
    bin_n = int(CONFIG["DECAY_BIN_S"] * sfreq)
    n_bins = int(CONFIG["DECAY_TOTAL_S"] / CONFIG["DECAY_BIN_S"])
    bin_centers = (np.arange(n_bins) + 0.5) * CONFIG["DECAY_BIN_S"]

    # focus on theta (offset effect) but compute all bands; store theta trajectory
    target_band = "theta"
    # trajectories[cond] -> list over subjects of (n_bins,) theta power
    traj = {c: [] for c in CONFIG["conditions"]}

    for subj in SUBJECTS:
        per_cond_ok = True
        subj_traj = {}
        for cond in CONFIG["conditions"]:
            # accumulate band power per time-bin, averaged over the 3 pauses
            bin_acc = [[] for _ in range(n_bins)]
            got = 0
            for block in CONFIG["noise_blocks"]:
                raw = load_raw(subj, block, cond, CONFIG["pause_file_tpl"])
                if raw is None:
                    continue
                data = raw.get_data()[:, lag:]
                got += 1
                for bi in range(n_bins):
                    seg = data[:, bi * bin_n:(bi + 1) * bin_n]
                    if seg.shape[1] < bin_n:
                        bin_acc[bi].append(np.nan); continue
                    bp = band_power(seg, sfreq, {target_band: FREQ_BANDS[target_band]},
                                    n_fft=min(bin_n, 64))
                    bin_acc[bi].append(bp[target_band])
            if got == 0:
                per_cond_ok = False; break
            subj_traj[cond] = np.array([np.nanmean(b) if b else np.nan for b in bin_acc])
        if not per_cond_ok:
            continue
        for cond in CONFIG["conditions"]:
            traj[cond].append(subj_traj[cond])

    if not traj["notch"]:
        print("  [skip] no pause data loaded"); return

    notch = np.vstack(traj["notch"])   # (n_subj, n_bins)
    white = np.vstack(traj["white"])

    # save
    pd.DataFrame(notch, columns=[f"t{c:.1f}" for c in bin_centers]).to_csv(
        os.path.join(CONFIG["outdir"], "offset_decay_notch_theta.csv"), index=False)
    pd.DataFrame(white, columns=[f"t{c:.1f}" for c in bin_centers]).to_csv(
        os.path.join(CONFIG["outdir"], "offset_decay_white_theta.csv"), index=False)

    # per-bin paired test notch vs white
    pvals = []
    for bi in range(n_bins):
        n_b = notch[:, bi]; w_b = white[:, bi]
        mask = ~(np.isnan(n_b) | np.isnan(w_b))
        if mask.sum() < 3:
            pvals.append(1.0); continue
        try:
            _, p = wilcoxon(n_b[mask], w_b[mask])
        except ValueError:
            p = 1.0
        pvals.append(p)
    reject = fdr_bh(pvals, CONFIG["fdr_alpha"])

    # plot trajectories with SEM
    plt.figure(figsize=(8, 5))
    for arr, lab, col in [(notch, "notch", "#378ADD"), (white, "white", "#888780")]:
        m = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
        plt.plot(bin_centers, m, "-o", color=col, label=lab)
        plt.fill_between(bin_centers, m - sem, m + sem, color=col, alpha=0.2)
    for bi, r in enumerate(reject):
        if r:
            plt.text(bin_centers[bi], plt.ylim()[1], "*", ha="center", fontsize=14)
    plt.xlabel("Time since offset (s)")
    plt.ylabel("Theta power (log PSD)")
    plt.title("Offset decay: theta power across the silent pause")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["outdir"], "offset_decay_theta.png"), dpi=150)
    plt.close()

    # quantify decay: slope of theta over time per condition (subject-wise)
    def slopes(arr):
        out = []
        for row in arr:
            mask = ~np.isnan(row)
            if mask.sum() >= 3:
                out.append(np.polyfit(bin_centers[mask], row[mask], 1)[0])
        return np.array(out)
    sn, sw = slopes(notch), slopes(white)
    try:
        _, p_slope = wilcoxon(sn[:min(len(sn), len(sw))], sw[:min(len(sn), len(sw))])
    except ValueError:
        p_slope = np.nan
    print(f"  theta decay slope: notch mean={sn.mean():.4f}, white mean={sw.mean():.4f}, "
          f"paired p={p_slope:.4f}")
    print(f"  per-bin notch>white significant bins (FDR): "
          f"{[f'{bin_centers[i]:.0f}s' for i,r in enumerate(reject) if r]}")
    print("  wrote offset_decay_theta.png")


# =====================================================================
# (3) Onset response and adaptation across the three blocks
# =====================================================================
def run_onset_adapt():
    print("\n=== (3) Onset response adaptation across blocks ===")
    sfreq = CONFIG["sfreq"]
    lo, hi = CONFIG["ONSET_WIN_S"]
    a, b = int(lo * sfreq), int(hi * sfreq)

    # onset response magnitude = mean rectified amplitude in the onset window,
    # averaged over channels, per block, per condition
    # store: rows = subject, cols = cond x block
    rows = []
    for subj in SUBJECTS:
        row = {"Subject": subj}
        ok = True
        for cond in CONFIG["conditions"]:
            for block in CONFIG["noise_blocks"]:
                raw = load_raw(subj, block, cond, CONFIG["noise_file_tpl"])
                if raw is None:
                    ok = False; break
                data = raw.get_data()[:, a:b]
                # mean absolute amplitude across channels & samples
                row[f"{cond}_b{block}"] = float(np.mean(np.abs(data)))
            if not ok:
                break
        if ok:
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CONFIG["outdir"], "onset_adaptation.csv"), index=False)
    if df.empty:
        print("  [skip] no subjects loaded"); return

    blocks = CONFIG["noise_blocks"]
    plt.figure(figsize=(7, 5))
    for cond, col in [("white", "#888780"), ("notch", "#378ADD")]:
        means = [df[f"{cond}_b{bk}"].mean() for bk in blocks]
        sems = [df[f"{cond}_b{bk}"].sem() for bk in blocks]
        plt.errorbar(blocks, means, yerr=sems, marker="o", color=col, label=cond,
                     capsize=4)
    plt.xticks(blocks); plt.xlabel("Block (repetition)")
    plt.ylabel("Onset response (mean |amplitude|)")
    plt.title("Onset response across repetitions")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["outdir"], "onset_adaptation.png"), dpi=150)
    plt.close()

    # Friedman test across blocks (adaptation) per condition
    for cond in CONFIG["conditions"]:
        cols = [df[f"{cond}_b{bk}"].values for bk in blocks]
        try:
            stat, p = friedmanchisquare(*cols)
            print(f"  {cond}: Friedman across blocks chi2={stat:.3f}, p={p:.4f}")
        except ValueError as e:
            print(f"  {cond}: Friedman failed ({e})")
    print("  wrote onset_adaptation.png")


# =====================================================================
# (5) Band-specific pause power over auditory channels (theta highlighted)
# =====================================================================
def run_band_auditory():
    print("\n=== (5) Pause-band power over auditory channels: notch vs white ===")
    sfreq = CONFIG["sfreq"]
    rows = []
    ch_names_ref = None
    for subj in SUBJECTS:
        per_cond = {}
        ok = True
        for cond in CONFIG["conditions"]:
            band_acc = {bd: [] for bd in FREQ_BANDS}
            for block in CONFIG["noise_blocks"]:
                raw = load_raw(subj, block, cond, CONFIG["pause_file_tpl"])
                if raw is None:
                    ok = False; break
                if ch_names_ref is None:
                    ch_names_ref = raw.ch_names
                # pick auditory channels that exist in this montage
                picks = [raw.ch_names.index(ch) for ch in raw.ch_names
                         if ch in raw.ch_names]
                # picks = raw.ch_names
                if not picks:
                    ok = False; break
                data = raw.get_data()[picks, :]
                perch = band_power_per_channel(
                    data, sfreq, FREQ_BANDS, n_fft=min(data.shape[1], 64))
                for bd in FREQ_BANDS:
                    band_acc[bd].append(np.nanmean(perch[bd]))
            if not ok:
                break
            per_cond[cond] = {bd: np.nanmean(band_acc[bd]) for bd in FREQ_BANDS}
        if not ok:
            continue
        row = {"Subject": subj}
        for bd in FREQ_BANDS:
            row[f"white_{bd}"] = per_cond["white"][bd]
            row[f"notch_{bd}"] = per_cond["notch"][bd]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(CONFIG["outdir"], "pause_band_auditory.csv"), index=False)
    if df.empty:
        print("  [skip] no pause data loaded"); return

    stats = []
    for bd in FREQ_BANDS:
        w = df[f"white_{bd}"].values; n = df[f"notch_{bd}"].values
        try:
            _, p = wilcoxon(n, w)
        except ValueError:
            p = 1.0
        # Cohen's d (paired)
        diff = n - w
        d = np.nanmean(diff) / (np.nanstd(diff, ddof=1) + 1e-12)
        stats.append({"band": bd, "white_mean": np.nanmean(w),
                      "notch_mean": np.nanmean(n),
                      "diff(notch-white)": np.nanmean(diff), "cohens_d": d, "p": p})
    sdf = pd.DataFrame(stats)
    sdf["p_fdr_reject"] = fdr_bh(sdf["p"].values, CONFIG["fdr_alpha"])

    print("fdr_pause", fdr_bh(sdf["p"].values, CONFIG["fdr_alpha"]))
    sdf.to_csv(os.path.join(CONFIG["outdir"], "pause_band_auditory_stats.csv"), index=False)
    print(sdf.round(4).to_string(index=False))

    bands = list(FREQ_BANDS)
    x = np.arange(len(bands)); wbar = 0.38
    wm = [df[f"white_{b}"].mean() for b in bands]
    nm = [df[f"notch_{b}"].mean() for b in bands]
    we = [df[f"white_{b}"].sem() for b in bands]
    ne = [df[f"notch_{b}"].sem() for b in bands]
    plt.figure(figsize=(4, 5))
    plt.bar(x - wbar/2, wm, wbar, yerr=we, capsize=4, label="white", color="#CC0000")
    plt.bar(x + wbar/2, nm, wbar, yerr=ne, capsize=4, label="notch", color="#0066CC")

    for i, b in enumerate(bands):
        if sdf.loc[sdf.band == b, "p_fdr_reject"].values[0]:
            plt.text(i, 0.08, "*", ha="center", fontsize=14)
    plt.xticks(x, ["δ", "θ", "α", "β", "γ"], fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("log PSD (mean over channels)", fontsize=14)
    plt.ylim(-1.4, 0.15)
    # plt.title("Pause-band power over auditory channels: notch vs white")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["outdir"], "pause_band_auditory.pdf"), dpi=150)
    plt.close()
    print("  wrote pause_band_auditory.png")


# =====================================================================
def main():
    if CONFIG["RUN_SUSTAINED"]:
        run_sustained()
    if CONFIG["RUN_OFFSET_DECAY"]:
        run_offset_decay()
    if CONFIG["RUN_ONSET_ADAPT"]:
        run_onset_adapt()
    if CONFIG["RUN_BAND_AUDITORY"]:
        run_band_auditory()
    print(f"\nDone. Outputs in '{CONFIG['outdir']}'.")


if __name__ == "__main__":
    main()
