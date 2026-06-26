import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.time_frequency import psd_array_welch

CONFIG = {
    "data_path": "...",
    "pause_file_tpl": "{block}_{subj}_{cond}_raw.fif",
    "noise_blocks": [1, 2, 3],
    "conditions": ["white", "notch"],     # 0 = white, 1 = notch
    "sfreq": 100,
    "outdir": "...",
    "PAUSE_OFFSET_LAG_S": 0.0,
    "data_scale": 1e6,                     # V -> uV
    "post_offset_win": (0.0, 0.8),         # unbiased full window
    "n_permutations": 5000,
    "cluster_alpha": 0.05,                 # cluster-forming threshold (p)
}
# FREQ_BANDS = {
#     "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
#     "beta": (13, 30), "gamma": (30, 45),
# }
THETA = (1,4)

# subjects
# _s = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]  #tinnitus
_s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 25, 26, 27, 29, 35, 38, 18, 19, 20, 23, 24]  #healthy

SUBJECTS = [f"Prob{i}" for i in _s]
os.makedirs(CONFIG["outdir"], exist_ok=True)


def load_pause(subj, block, cond):
    fp = os.path.join(CONFIG["data_path"],
                      CONFIG["pause_file_tpl"].format(block=block, subj=subj, cond=cond))
    return mne.io.read_raw_fif(fp, verbose=False) if os.path.exists(fp) else None


def theta_per_channel(data, sfreq):
    n_fft = min(data.shape[1], 64)
    psd, freqs = psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45,
                                 n_fft=n_fft, verbose=False)
    psd = np.log10(psd + 1e-20)
    idx = np.logical_and(freqs >= THETA[0], freqs < THETA[1])
    return psd[:, idx].mean(axis=1)


def fdr_bh(pvals, alpha=0.05):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    passed = ranked <= (np.arange(1, n + 1) / n) * alpha
    reject = np.zeros(n, bool)
    if passed.any():
        reject[order[:np.max(np.where(passed)[0]) + 1]] = True
    return reject


def main():
    sfreq = CONFIG["sfreq"]
    lag = int(CONFIG["PAUSE_OFFSET_LAG_S"] * sfreq)
    t0, t1 = CONFIG["post_offset_win"]
    a, b = int(t0 * sfreq), int(t1 * sfreq)

    info = None
    ch_names = None
    # per-subject theta per channel, per condition (averaged over 3 pauses)
    white_mat, notch_mat = [], []

    for subj in SUBJECTS:
        cond_vals = {}
        ok = True
        for cond in CONFIG["conditions"]:
            acc, got = [], 0
            for block in CONFIG["noise_blocks"]:
                raw = load_pause(subj, block, cond)
                if raw is None:
                    continue
                if info is None:
                    info = raw.info
                    ch_names = raw.ch_names
                got += 1
                data = raw.get_data() * CONFIG["data_scale"]
                data = data[:, lag:][:, a:b]
                if data.shape[1] < 8:
                    continue
                acc.append(theta_per_channel(data, sfreq))
            if got == 0 or not acc:
                ok = False; break
            cond_vals[cond] = np.mean(acc, axis=0)
        if not ok:
            continue
        white_mat.append(cond_vals["white"])
        notch_mat.append(cond_vals["notch"])

    white_mat = np.vstack(white_mat)   # (n_subj, n_ch)
    notch_mat = np.vstack(notch_mat)
    diff = notch_mat - white_mat       # notch - white, per subject per channel
    n_subj, n_ch = diff.shape
    print(f"Loaded {n_subj} subjects, {n_ch} channels.")

    # ---------- (A) cluster-based permutation over all channels ----------
    print("\n=== Cluster-based permutation test (all channels) ===")
    try:
        from mne.stats import spatio_temporal_cluster_1samp_test
        from mne.channels import find_ch_adjacency

        # adjacency from the montage geometry
        adjacency, adj_names = find_ch_adjacency(info, ch_type="eeg")
        X = diff[:, np.newaxis, :]
        from scipy.stats import t as tdist
        thresh = tdist.ppf(1 - CONFIG["cluster_alpha"] / 2, n_subj - 1)
        T_obs, clusters, clu_p, _ = spatio_temporal_cluster_1samp_test(
            X, threshold=thresh, n_permutations=CONFIG["n_permutations"],
            adjacency=adjacency, n_jobs=1, seed=42, verbose=False)
        sig = [(i, p) for i, p in enumerate(clu_p) if p < 0.05]
        print("sig", sig)
        print("T_obs", T_obs)
        print("clu_p", clu_p)
        if sig:
            for i, p in sig:
                chs = np.unique(clusters[i][1])
                names = [ch_names[c] for c in chs]
                print(clusters, ch_names)
                print(f"  cluster {i}: p={p:.4f}, {len(names)} channels: {names}")
        else:
            print(f"  no cluster below p=0.05 (min cluster p="
                  f"{min(clu_p) if len(clu_p) else float('nan'):.4f})")
        sig_channels = set()
        for i, p in sig:
            sig_channels.update(np.unique(clusters[i][1]))
    except Exception as e:
        print(f"  [cluster test skipped: {e}]")
        sig_channels = set()
    # ---------- topographic map of the difference ----------

    try:
        mean_diff = diff.mean(axis=0)   # (n_ch,)
        fig, ax = plt.subplots(figsize=(5, 5))
        mask = np.zeros(n_ch, dtype=bool)
        mask[list(sig_channels)] = True
        mne.viz.plot_topomap(
            mean_diff, info, axes=ax, show=False, cmap="RdBu_r",
            mask=mask if mask.any() else None,
            mask_params=dict(marker="o", markerfacecolor="k", markersize=6))
        # ax.set_title("Notch - White theta (post-offset)\nmarked = cluster-significant")
        fig.tight_layout()
        fig.savefig(os.path.join(CONFIG["outdir"], "stim3_sustained_topomap.pdf"), dpi=200)
        plt.close(fig)
        print("\n  wrote stim3_theta_topomap.png")
    except Exception as e:
        print(f"  [topomap skipped: {e}]")


if __name__ == "__main__":
    main()
