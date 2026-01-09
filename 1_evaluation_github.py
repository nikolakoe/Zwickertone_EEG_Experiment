import mne
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro

data_1_n = []
data_1_w = []
data_2_n = []
data_2_w = []
data_3_n = []
data_3_w = []

# subjects grouped by classes (ZT = Zwickertone perceived, T = Tinnitus subjects, bN = bigger Notch)
subjects_ZT = [1, 4, 10, 11, 16]
subjects_T = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]
subjects_bN = [18, 19, 20, 23, 24]
subjects = [2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17, 25, 26, 27, 29, 35, 38]
subjects_ = subjects + subjects_T + subjects_ZT
for i in subjects_:

    data_2378_n = mne.read_evokeds(".../1_Prob_" + str(i) + "_2378_notch_ave.fif")[0]
    data_2378_w = mne.read_evokeds(".../1_Prob_" + str(i) + "_2378_white_ave.fif")[0]
    data_2828_n = mne.read_evokeds(".../1_Prob_" + str(i) + "_2828_notch_ave.fif")[0]
    data_2828_w = mne.read_evokeds(".../1_Prob_" + str(i) + "_2828_white_ave.fif")[0]
    data_3364_n = mne.read_evokeds(".../1_Prob_" + str(i) + "_3364_notch_ave.fif")[0]
    data_3364_w = mne.read_evokeds(".../1_Prob_" + str(i) + "_3364_white_ave.fif")[0]

    data_1_n.append(data_2378_n)
    data_1_w.append(data_2378_w)
    data_2_n.append(data_2828_n)
    data_2_w.append(data_2828_w)
    data_3_n.append(data_3364_n)
    data_3_w.append(data_3364_w)

# temporal channels and time range for statistical test
picks=["FT10", "FT8", "T8", "TP8", "TP10", "FT9", "FT7", "T7", "TP7", "TP9"]
tmin=0.0
tmax=0.1
d1_n = []
d1_w = []
d2_n = []
d2_w = []
d3_n = []
d3_w = []

for i in range(len(data_1_n)):
    d1_n.append(np.mean(data_1_n[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    d1_w.append(np.mean(data_1_w[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    d2_n.append(np.mean(data_2_n[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    d2_w.append(np.mean(data_2_w[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    d3_n.append(np.mean(data_3_n[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))
    d3_w.append(np.mean(data_3_w[i].get_data(picks=picks, tmin=tmin, tmax=tmax)))

def test_(d_n, d_w):
    # Example: RMS amplitudes for 38 subjects in two conditions
    cond1 = np.array(d_n)  # e.g., white noise
    cond2 = np.array(d_w)  # e.g., notch noise

    # --- 1. Check normality of the difference ---
    diff = cond1 - cond2
    stat, p_norm = shapiro(diff)
    #print(f"Shapiro-Wilk normality p = {p_norm:.4f}")

    # --- 2. Choose the right test ---
    if p_norm > 0.05:
        # differences are approximately normal → paired t-test
        t_stat, p_val = ttest_rel(cond1, cond2)
        print(f"Paired t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
    else:
        # non-normal → use Wilcoxon signed-rank test
        w_stat, p_val = wilcoxon(cond1, cond2)
        print(f"Wilcoxon signed-rank: W = {w_stat:.3f}, p = {p_val:.4f}")

# test results for the three different tones
test_(d1_n, d1_w)
test_(d2_n, d2_w)
test_(d3_n, d3_w)

