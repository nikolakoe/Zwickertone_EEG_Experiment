import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro


data_n_on = []
data_n_off = []
data_w_on = []
data_w_off = []

# subjects grouped by classes (ZT = Zwickertone perceived, T = Tinnitus subjects, bN = bigger Notch)
subjects_ZT = [1, 4, 10, 11, 16]
subjects_T = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]
subjects_bN = [18, 19, 20, 23, 24]
subjects = [2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17, 25, 26, 27, 29, 35, 38]

subjects_ = subjects_ZT + subjects_T + subjects


path = ".../"

diffs = []
for i in subjects_:
    data_n_onset = mne.read_evokeds(path + "4_Prob" + str(i) + "_notch_onset_ave.fif")[0]
    data_n_offset = mne.read_evokeds(path + "4_Prob" + str(i) + "_notch_offset_ave.fif")[0]
    data_w_onset = mne.read_evokeds(path + "4_Prob" + str(i) + "_white_onset_ave.fif")[0]
    data_w_offset = mne.read_evokeds(path + "4_Prob" + str(i) + "_white_offset_ave.fif")[0]

    data_n_on.append(data_n_onset)
    data_n_off.append(data_n_offset)
    data_w_on.append(data_w_onset)
    data_w_off.append(data_w_offset)


# picks=["FT10", "FT8", "T8", "TP8", "TP10"] #right temporal channels
picks=["FT9", "FT7", "T7", "TP7", "TP9"]    #left temporal channels
tmin = 0.2
tmax = 0.4

data_n_on_ = mne.combine_evoked(data_n_on, weights="nave")
data_n_off_ = mne.combine_evoked(data_n_off, weights="nave")
data_w_on_ = mne.combine_evoked(data_w_on, weights="nave")
data_w_off_ = mne.combine_evoked(data_w_off, weights="nave")


with mne.viz.use_browser_backend("matplotlib"):
    data_n_off_.plot_topomap(times=[0.3], average=0.2, show=True, vlim=(-0.7, 0.7))
    plt.savefig(".../data_n_off_topo.pdf", dpi=200, bbox_inches="tight")

n_on=np.mean(data_n_on_.get_data(picks=picks), axis=0)
n_off=np.mean(data_n_off_.get_data(picks=picks), axis=0)
w_on=np.mean(data_w_on_.get_data(picks=picks), axis=0)
w_off=np.mean(data_w_off_.get_data(picks=picks), axis=0)


x = np.arange(len(n_on))/100 - 0.2

# plotting ERPs
plt.figure(figsize=(8,6))
plt.subplot(1, 2, 1)
plt.hlines(0, -0.3, 1, linestyles="dashed", linewidth=2, color="darkgrey")
plt.plot(x, n_on*1000000, color="#0066CC", label="notch onset", linewidth=3)
plt.plot(x, w_on*1000000, color="#CC0000", label="white onset", linewidth=3)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Activity [μV]", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.subplot(1,2,2)
plt.hlines(0, -0.3, 1, linestyles="dashed", linewidth=2, color="darkgrey")
plt.plot(x, n_off*1000000, color="#0066CC", label="notch offset", linewidth=3)
plt.plot(x, w_off*1000000, color="#CC0000", label="white offset", linewidth=3)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Activity [μV]", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()


##### statistical testing

n_on = []
n_off = []
w_on = []
w_off = []

# calculate mean activity in time frame
for i in range(len(data_n_on)):
    n_on.append(np.mean(data_n_on[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    n_off.append(np.mean(data_n_off[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    w_on.append(np.mean(data_w_on[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    w_off.append(np.mean(data_w_off[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))


plt.bar(["n_off", "w_off"], [np.mean(np.array(n_off)), np.mean(np.array(w_off))], yerr=[np.std(np.array(n_off)), np.std(np.array(w_off))])
plt.show()


# offset responses

cond1 = np.array(n_off)  # e.g., notch noise
cond2 = np.array(w_off)  # e.g., white noise

# --- 1. Check normality of the difference ---
diff = cond1 - cond2
stat, p_norm = shapiro(diff)
print(f"Shapiro-Wilk normality p = {p_norm:.4f}")

# --- 2. Choose the right test ---
if p_norm > 0.05:
    # differences are approximately normal → paired t-test
    t_stat, p_val = ttest_rel(cond1, cond2)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
else:
    # non-normal → use Wilcoxon signed-rank test
    w_stat, p_val = wilcoxon(cond1, cond2)
    print(f"Wilcoxon signed-rank: W = {w_stat:.3f}, p = {p_val:.4f}")

# onset responses

cond1 = np.array(n_on)  # e.g., notch noise
cond2 = np.array(w_on)  # e.g., white noise

# --- 1. Check normality of the difference ---
diff = cond1 - cond2
stat, p_norm = shapiro(diff)
print(f"Shapiro-Wilk normality p = {p_norm:.4f}")

# --- 2. Choose the right test ---
if p_norm > 0.05:
    # differences are approximately normal → paired t-test
    t_stat, p_val = ttest_rel(cond1, cond2)
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
else:
    # non-normal → use Wilcoxon signed-rank test
    w_stat, p_val = wilcoxon(cond1, cond2)
    print(f"Wilcoxon signed-rank: W = {w_stat:.3f}, p = {p_val:.4f}")

