import mne
import numpy as np
from scipy.stats import ttest_rel
import random
from mne.stats import fdr_correction

random.SEED = 42
np.random = 42

tmax = 0.8

epochsN = []
epochsW = []
path = ".../4_Prob"

# subjects grouped by classes (ZT = Zwickertone perceived, T = Tinnitus subjects, bN = bigger Notch)
subjects_ZT = [1, 4, 10, 11, 16]
subjects_T = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]
subjects_bN = [18, 19, 20, 23, 24]
subjects = [2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17, 25, 26, 27, 29, 35, 38]

subjects_ = subjects_ZT + subjects_T + subjects + subjects_bN

for i in subjects_:
    if i ==21:
        continue

    n = mne.read_epochs(path + str(i) + "_notch_offset_epo.fif").crop(tmin=0, tmax=tmax)
    w = mne.read_epochs(path + str(i) + "_white_offset_epo.fif").crop(tmin=0, tmax=tmax)
    info = n.info
    sfreq = n.info["sfreq"]

    epochsN.append(n)
    epochsW.append(w)


fmin, fmax = 1, 45
n_fft = int(tmax*sfreq)

psdN = []
psdW = []

for epochsN, epochsW in zip(epochsN, epochsW):
    print(epochsN)
    pN, freqs = mne.time_frequency.psd_array_welch(epochsN.get_data(), fmin=fmin, fmax=fmax, n_fft=n_fft, sfreq=sfreq)
    pW, _ = mne.time_frequency.psd_array_welch(epochsW.get_data(), fmin=fmin, fmax=fmax, n_fft=n_fft, sfreq=sfreq)

    # average across epochs → (n_channels, n_freqs)
    psdN.append(pN.mean(axis=0))
    psdW.append(pW.mean(axis=0))

psdN = np.stack(psdN)  # shape (n_subj, n_ch, n_freqs)
psdW = np.stack(psdW)

psdN_db = 10 * np.log10(psdN)
psdW_db = 10 * np.log10(psdW)

bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 12),
    'beta':  (13, 30),
    'gamma': (30, 45)
}


def band_power_per_channel(psd_db, freqs, fmin, fmax):
    """Return mean power per channel, averaged across frequency range, per subject"""
    idx = (freqs >= fmin) & (freqs <= fmax)
    return psd_db[..., idx].mean(axis=-1)  # shape: (n_subjects, n_channels)

def band_power(psd_db, freqs, fmin, fmax):
    idx = (freqs >= fmin) & (freqs <= fmax)
    # mean over frequencies *and channels* (whole-brain summary)
    return psd_db[..., idx].mean(axis=(-1, -2))

band_results = {}

for name, (lo, hi) in bands.items():
    bpN = band_power(psdN_db, freqs, lo, hi)  # shape (n_subjects,)
    bpW = band_power(psdW_db, freqs, lo, hi)
    band_results[name] = (bpN, bpW)

print("shape", bpN.shape, bpW.shape)

p_fdrs = []
for name, (bpN, bpW) in band_results.items():
    t, p = ttest_rel(bpN, bpW)  # condB − condA
    p_fdrs.append(p)
    diff = (bpW - bpN).mean()
    print(f"{name:6s}  t = {t:.3f},   p = {p:.4f},   mean difference = {diff:.3f} dB")

print(fdr_correction(p_fdrs))
def cohens_d(a, b):
    return (b - a).mean() / (b - a).std(ddof=1)

for name, (bpN, bpW) in band_results.items():
    print("cohen", name, cohens_d(bpN, bpW))


band_diffs = {name: (bpW - bpN).mean() for name, (bpN, bpW) in band_results.items()}
print(band_diffs)
print("Largest difference in:", max(band_diffs, key=lambda k: abs(band_diffs[k])))
