import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# load files generated with "Zwickertone_Evaluation_github.py"
am_with_notch = np.load("AM_with_notch.npy")
am_without_notch = np.load("AM_without_notch.npy")
freqs = np.load("freqs.npy")

with_ = []
without_ = []

# subjects grouped by classes (ZT = Zwickertone perceived, T = Tinnitus subjects, bN = bigger Notch)
subjects_ZT = [1, 4, 10, 11, 16]
subjects_T = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]
subjects_bN = [18, 19, 20, 23, 24]
subjects = [2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17, 25, 26, 27, 29, 35, 38]

subjects_ordered = []
p_values_ordered = []
colors = []

subjects_smallerNotch = subjects_ZT + subjects_T + subjects

am_with = []
am_without = []
for i in subjects_smallerNotch:
    if i > 20:
        i-=1
    with_.append(am_with_notch[i-1][freqs==40])
    without_.append(am_without_notch[i-1][freqs==40])
    am_with.append(am_with_notch[i-1])
    am_without.append(am_without_notch[i-1])

# Example data
cond_A = np.array(with_)
cond_B = np.array(without_)

# Check normality of differences
diff = cond_B - cond_A
# print(stats.shapiro(diff))
stat, p_norm = stats.shapiro(diff)
print(f"Shapiro-Wilk normality p = {p_norm:.4f}")

# --- 2. Choose the right test ---
if p_norm > 0.05:
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(cond_A, cond_B)
    print("t =", t_stat, "p =", p_val)

# OR if not normal
w_stat, p_val = stats.wilcoxon(cond_A, cond_B)
print("Wilcoxon W =", w_stat, "p =", p_val)

psds_with_notch = np.load("psds_w_n.npy")
psds_without_notch = np.load("psds_w_no.npy")

w = []
wo = []
for i in subjects_smallerNotch:
    if i > 20:
        i-=1
    a = psds_with_notch[i-1][:,freqs==40]*10**14
    print(a)
    a = [i / np.max(np.abs(a)) for i in a]
    print(a)
    w.append(a)
    b = psds_without_notch[i-1][:,freqs==40]
    b = [i / np.max(np.abs(b)) for i in b]
    wo.append(b)

w = np.mean(w, axis=0)
wo = np.mean(wo, axis=0)

# plot psd
plt.figure(figsize=(3,4))
plt.vlines(40, 0, 1, linestyle="dashed", color="orange", linewidth=3, alpha=0.8)
plt.plot(freqs, np.mean(am_with, axis=0), color="#004C99", linewidth=3)
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("RMS Power", fontsize=14)
plt.xlim(32, 48)  # Zoom into 40 Hz
plt.ylim(0, 0.4 * 10 ** (-11))
plt.xticks(ticks=[35, 40, 45], fontsize=14)
plt.yticks(ticks=[0,0.1 * 10 ** (-11),0.2 * 10 ** (-11),0.3 * 10 ** (-11),0.4 * 10 ** (-11)], fontsize=14)
plt.savefig(".../AM_with_notch_smallerNotch.pdf", dpi=200, bbox_inches="tight")
plt.show()
plt.figure(figsize=(3,4))
plt.vlines(40, 0, 1, linestyle="dashed", color="orange", linewidth=3, alpha=0.8)
plt.plot(freqs, np.mean(am_without, axis=0), color="grey", linewidth=3)
plt.xlabel("Frequency (Hz)", fontsize=14)
plt.ylabel("RMS Power", fontsize=14)
plt.xlim(32, 48)  # Zoom into 40 Hz
plt.xticks(ticks=[35, 40, 45], fontsize=14)
plt.yticks(ticks=[0,0.1 * 10 ** (-11),0.2 * 10 ** (-11),0.3 * 10 ** (-11),0.4 * 10 ** (-11)], fontsize=14)
plt.ylim(0, 0.4 * 10 ** (-11))
plt.savefig(".../AM_without_notch_smallerNotch.pdf", dpi=200, bbox_inches="tight")
plt.show()