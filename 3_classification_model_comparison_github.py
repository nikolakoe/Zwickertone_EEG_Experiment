import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import random
import os
import mne
from mne.time_frequency import psd_array_welch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from sklearn.metrics import accuracy_score
from keras import models, layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score

# Define random seed for reproducibility
SEED = 42
# Python built-in random module
random.seed(SEED)
# NumPy random
np.random.seed(SEED)


# Define models to compare
models = {
    "LogReg": LogisticRegression(C=0.1, max_iter=1000, random_state=SEED),
    "SVM_": SVC(kernel="rbf", C=1, gamma="scale", random_state=SEED),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, max_features="sqrt", random_state=SEED),
    "Dense": models.Sequential([
        layers.GaussianNoise(0.01, input_shape=(315,)),
        layers.Dense(64, activation='tanh', input_shape=(315,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ]),
   "Conv": models.Sequential([
        layers.GaussianNoise(0.01, input_shape=(63, 5, 1)),          # regularization
        layers.Conv2D(32, (5,1), activation='relu', padding='same',  # spatial filters
               kernel_regularizer=l2(1e-3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,1)),                                  # spatial downsampling
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(32, activation='relu', kernel_regularizer=l2(1e-3)),
        layers.Dense(1, activation='sigmoid')
    ])
}

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
s = np.concatenate((np.arange(1,21), np.arange(22,40))) # leave out subject 21 due to neurological disease
subjects = [f"Prob{i}" for i in s]
conditions = ["white", "notch"]
sfreq = 100          # Hz
window_len = 2       # seconds
win_samp = sfreq * window_len  # 500 samples per window

def load_data(subject, condition):
    fpath = os.path.join(data_path, f"{subject}_{condition}.npy")
    return np.load(fpath)  # shape (n_channels, n_samples)

def segment_data(eeg, window_samples):
    n_ch, n_samp = eeg.shape
    n_windows = int(n_samp // window_samples)
    window_samples = int(window_samples)
    return np.stack(
        [eeg[:, i * window_samples:(i + 1) * window_samples] for i in range(n_windows)],
        axis=0
    )  # shape: (n_windows, n_ch, window_samples)

def compute_bandpower_features(segment, sfreq, bands):
    psd, freqs = psd_array_welch(segment, sfreq=sfreq, fmin=1, fmax=45, n_fft=200, verbose=False)  # 256
    features = []
    for fmin, fmax in bands.values():
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        band_power = psd[:, idx].mean(axis=1)
        features.append(band_power)
    return np.concatenate(features)  # (n_channels * n_bands,)

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

def compute_psd_features_1hz(segment, sfreq):
    """
    Compute log-transformed PSD features averaged in 1 Hz bins (1–45 Hz).
    Returns a feature vector of shape (n_channels * n_freq_bins,).
    """
    # Step 1: Compute PSD
    psd, freqs = psd_array_welch(
        segment, sfreq=sfreq, fmin=1, fmax=45, n_fft=200, verbose=False
    )

    # Step 2: Log-transform to stabilize variance
    psd = np.log10(psd + 1e-10)

    # Step 3: Define 1 Hz bin edges (1–46 gives 45 bins)
    freq_bins = np.arange(1, 46, 7)

    # Step 4: Compute mean power per 1 Hz bin
    features = []
    for fmin, fmax in zip(freq_bins[:-1], freq_bins[1:]):
        idx = np.logical_and(freqs >= fmin, freqs < fmax)
        if np.any(idx):
            bin_power = psd[:, idx].mean(axis=1)
        else:
            # If no frequency falls into this bin (shouldn't happen with Welch), fill with NaN or zeros
            bin_power = np.zeros(psd.shape[0])
        features.append(bin_power)

    # Step 5: Concatenate across bins → (n_channels * n_bins)
    return np.concatenate(features)


all_results = []
y_trues = []
y_preds = []

for subj in subjects:
    print(f"\n🧠 Processing {subj}")

    # Load both conditions
    X_train, y_train = [], []
    X_test, y_test = [], []
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
        segments_train = np.concatenate((segments1, segments2))
        segments_test = segments3
        for seg in segments_train:
            feats = compute_bandpower_features_log(seg, sfreq, freq_bands)
            X_train.append(feats)
            y_train.append(ci)
        for seg in segments_test:
            feats = compute_bandpower_features_log(seg, sfreq, freq_bands)
            X_test.append(feats)
            y_test.append(ci)

    X_tr = np.array(X_train)
    y_tr = np.array(y_train)
    X_te = np.array(X_test)
    y_te = np.array(y_test)
    X_tr = StandardScaler().fit_transform(X_tr)
    X_te = StandardScaler().fit_transform(X_te)

    X_tr, y_tr = shuffle(X_tr, y_tr, random_state=42)
    X_te, y_te = shuffle(X_te, y_te, random_state=42)

    subj_results = {"Subject": subj}

    for name, model in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        if name == "Dense" or name=="Dense_" or name=="Dense__" or name=="Dense___":
            model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
            es = EarlyStopping(patience=10, restore_best_weights=True)
            history = model.fit(X_tr, y_tr, validation_split=0.2, epochs=200, batch_size=8, callbacks=[es], verbose=1)
            y_pred_prob = model.predict(X_te)
            y_pred = (y_pred_prob > 0.5).astype(int)
            acc = accuracy_score(y_te, y_pred)
        elif name == "Conv" or name=="Conv_" or name=="Conv__" or name=="Conv___":
            X_cnn = X_tr.reshape(X_tr.shape[0], 63, 5, 1)
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            es = EarlyStopping(patience=10, restore_best_weights=True)

            history = model.fit(
                X_cnn, y_tr,
                validation_split=0.2,
                epochs=200,
                batch_size=8,
                callbacks=[es],
                verbose=1
            )
            X_cnn_test = X_te.reshape(X_te.shape[0], 63, 5, 1)
            y_pred_prob = model.predict(X_cnn_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            acc = accuracy_score(y_te, y_pred)
        else:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
        subj_results[name] = np.mean(acc)

        if name=="LogReg":
            y_trues.append(y_te)
            y_preds.append(y_pred)

    all_results.append(subj_results)


auc_scores = []
for subj, yt, yp in zip(subjects, y_trues, y_preds):
    auc = roc_auc_score(yt,yp)
    auc_scores.append(auc)

sns.set(style="whitegrid")

plt.figure(figsize=(6, 4))
sns.violinplot(data=auc_scores, inner='point', color='skyblue')
plt.axhline(0.5, ls='--', color='gray', label='Chance level')
plt.ylabel('AUC')
plt.title('Decoding performance across participants')
plt.legend()
plt.show()

df_results = pd.DataFrame(all_results)
df_results.to_csv("multi_model_comparison_subjects.csv", index=False)
print("\nSaved results to model_comparison_subjects.csv")

# -------------------------------
# 6. Compute mean ± std across subjects
# -------------------------------
summary = df_results.drop(columns="Subject").agg(['mean', 'std']).T
print("\n=== Mean accuracies across subjects ===")
print(summary)

# -------------------------------
# 7. Visualization
# -------------------------------
plt.figure(figsize=(8, 5))
plt.bar(summary.index, summary["mean"], yerr=summary["std"], capsize=5)
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Model Comparison Across Subjects")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Melt DataFrame into long format for plotting
df_long = df_results.melt(id_vars="Subject", var_name="Model", value_name="Accuracy")

plt.figure(figsize=(9, 5))
sns.pointplot(
    data=df_long,
    x="Model", y="Accuracy",
    estimator=np.mean, errorbar='sd',
    color='black', linestyle="none", capsize=0.1
)

# Overlay each subject’s accuracies as paired lines
for subj in df_results["Subject"]:
    subj_data = df_long[df_long["Subject"] == subj]
    plt.plot(subj_data["Model"], subj_data["Accuracy"], alpha=0.4, linewidth=1)

plt.title("Subject-wise Classification Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Sort subjects to ensure consistent order
subjects_sorted = sorted(df_results["Subject"])

# Create subplots — adjust grid size depending on number of subjects
n_subjects = len(subjects_sorted)
n_cols = 5  # you can change this (e.g., 5 columns per row)
n_rows = math.ceil(n_subjects / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), sharey=True)
axes = axes.flatten()

for i, subj in enumerate(subjects_sorted):
    ax = axes[i]
    subj_data = df_results[df_results["Subject"] == subj].drop(columns="Subject").T
    subj_data.columns = ["Accuracy"]
    subj_data.plot(kind="bar", ax=ax, legend=False, color="skyblue", edgecolor="black")

    ax.set_title(f"{subj}")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(subj_data.index, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylabel("Accuracy" if i % n_cols == 0 else "")

# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Model Accuracies per Subject", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

