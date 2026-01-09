import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import random
import os
import mne
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from itertools import combinations
from scipy.spatial.distance import cdist


SEED = 42
# Python built-in random module
random.seed(SEED)
# NumPy random
np.random.seed(SEED)

freq_bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
}

# Prepare cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

data_path = ".../"  # folder containing preprocessed EEG files
s = np.concatenate((np.arange(1,21), np.arange(22,40))) # remove subject 21 due to neurological disease
subjects = [f"Prob{i}" for i in s]
conditions = ["notch", "white"]
sfreq = 100          # Hz
window_len = 2       # 0.5 seconds for pauses
win_samp = sfreq * window_len


def compute_bandpower_features_log(segment, sfreq, bands):
    # Compute PSD (linear units)
    psd, freqs = psd_array_welch(
        segment, sfreq=sfreq, fmin=1, fmax=45, n_fft=50, verbose=False
    )

    # Step 1: Log-transform PSD (add epsilon to avoid log(0))
    psd = np.log10(psd + 1e-10)

    # Step 2: Compute mean log-power in each band
    features = []
    for fmin, fmax in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        band_power = psd[:, idx].mean(axis=1)  # mean across frequencies within the band
        features.append(band_power)

    # Step 3: Concatenate features (channels × bands)
    return np.concatenate(features)

def segment_data(eeg, window_samples):
    n_ch, n_samp = eeg.shape
    n_windows = int(n_samp // window_samples)
    window_samples = int(window_samples)
    return np.stack(
        [eeg[:, i * window_samples:(i + 1) * window_samples] for i in range(n_windows)],
        axis=0
    )  # shape: (n_windows, n_ch, window_samples)


def global_distance_variance(X, y):
    classes = np.unique(y)
    n_classes = len(classes)

    # Compute within-class distances
    within_dists = []
    for c in classes:
        Xc = X[y == c]
        if len(Xc) > 1:
            dists = cdist(Xc, Xc)
            within_dists.append(np.mean(dists))
    d_within = np.mean(within_dists)

    # Compute between-class distances
    between_dists = []
    for c1, c2 in combinations(classes, 2):
        X1, X2 = X[y == c1], X[y == c2]
        dists = cdist(X1, X2)
        between_dists.append(np.mean(dists))
    d_between = np.mean(between_dists)

    # GDV formula
    gdv = (d_between - d_within) / (d_between + d_within)
    return gdv


gdvs = []
for subj in subjects:
    print(f"\n🧠 Processing {subj}")
    X, y = [], []
    # Load both conditions
    for ci, cond in enumerate(conditions):
        fpath = os.path.join(data_path, f"1_{subj}_{cond}_raw.fif")
        eeg1 = mne.io.read_raw_fif(fpath).get_data()
        segments1 = segment_data(eeg1, win_samp)
        fpath = os.path.join(data_path, f"2_{subj}_{cond}_raw.fif")
        eeg2 = mne.io.read_raw_fif(fpath).get_data()
        segments2 = segment_data(eeg2, win_samp)
        fpath = os.path.join(data_path, f"3_{subj}_{cond}_raw.fif")
        eeg3 = mne.io.read_raw_fif(fpath).get_data()
        segments3 = segment_data(eeg3, win_samp)
        print(segments1.shape, segments2.shape, segments3.shape)
        segments = np.concatenate((segments1, segments2, segments3))
        for seg in segments:
            feats = compute_bandpower_features_log(seg, sfreq, freq_bands)
            X.append(feats)
            y.append(ci)

    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=y, cmap='coolwarm', s=60, alpha=0.8, edgecolor='k')

    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["Notch Noise Pauses", "White Noise Pauses"], fontsize=16)

    plt.tight_layout()
    plt.savefig(".../Pause_Cluster_plot_" + subj + "_pause.pdf", dpi=200, bbox_inches="tight")
    plt.show()

print(np.mean(gdvs), np.std(gdvs))
