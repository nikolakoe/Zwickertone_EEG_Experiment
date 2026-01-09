import numpy as np
import mne
from mne.datasets import fetch_fsaverage, sample
import os.path as op
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.io.wavfile import read
import pandas as pd
from scipy.signal import welch


def finding_shift(protocol, proband_stim):
    time = len(protocol[1]) / 96000
    data_sample = np.interp(np.arange(0, len(protocol[1]), (len(protocol[1]) / time) / fsamp),
                            np.arange(0, len(protocol[1])), protocol[1])

    # correlate stimuli channel with stimuli protocol to find the time shift and thus start of the clicks
    corr = correlate(proband_stim['65'], data_sample, "valid")

    sorted_indices = np.argsort(np.abs(corr))[::-1]

    # First: index of the highest value
    first_idx = sorted_indices[0]
    second_idx = 0
    # Second: find the next highest index that's at least 100 away
    for idx in sorted_indices:
        if abs(idx - first_idx) > 100:
            second_idx = idx
            break

    shifts = np.sort((first_idx, second_idx))

    # the min value of the correlation depicts the shift index
    # get the time for the shift index
    shift_time1 = proband_stim['time'].iloc[shifts[0]]
    shift_time2 = proband_stim['time'].iloc[shifts[1]]

    # plot the correlation vor verification
    # plt.figure(figsize=(10, 3))
    # plt.subplot(2, 2, 1)
    # x = np.arange(len(corr)) / fsamp
    # plt.plot(x, corr)
    # plt.xlabel("Shift [s]")
    # plt.ylabel("Correlation")
    # # plot the match
    # plt.subplot(2, 2, 2)
    # data_shift = [None] * len(proband_stim['65'])
    # data_shift[shifts[0]:shifts[0] + len(data_sample)] = data_sample / 3
    # x = np.arange(len(proband_stim['65'])) / fsamp
    # plt.plot(x, proband_stim['65'], label="Stimuli Channel")
    # plt.plot(x, data_shift, label="Stimuli Protocol")
    # plt.xlim(shift_time1 - 3, shift_time1 + 40)
    # plt.xlabel("Time [s]")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # x = np.arange(len(corr)) / fsamp
    # plt.plot(x, corr)
    # plt.xlabel("Shift [s]")
    # plt.ylabel("Correlation")
    # # plot the match
    # plt.subplot(2, 2, 4)
    # data_shift = [None] * len(proband_stim['65'])
    # data_shift[shifts[1]:shifts[1] + len(data_sample)] = data_sample / 3
    # x = np.arange(len(proband_stim['65'])) / fsamp
    # plt.plot(x, proband_stim['65'], label="Stimuli Channel")
    # plt.plot(x, data_shift, label="Stimuli Protocol")
    # plt.xlim(shift_time2 - 3, shift_time2 + 40)
    # plt.xlabel("Time [s]")
    # plt.legend()
    # plt.show()

    return shift_time1, shift_time2


def get_fft(eeg, file):
    data_, sfreq = eeg.get_data(return_times=False), eeg.info['sfreq']  # shape: (n_channels, n_times)

    # Parameters for Welch
    nperseg = 300000  # 32768 * 15  # Try 4096 or higher if sampling rate allows
    noverlap = nperseg // 2

    # Compute PSDs for each channel
    psds = []
    freqs = None
    for ch in data_:
        f, p = welch(ch, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
        psds.append(p)
        if freqs is None:
            freqs = f
    psds = np.array(psds)  # shape: (n_channels, n_freqs)

    # Compute RMS over all channels
    rms_spectrum = np.sqrt(np.mean(psds ** 2, axis=0))

    # Plot
    plt.figure()
    plt.vlines(40, 0, 1, linestyle="dashed", color="orange")
    plt.plot(freqs, rms_spectrum)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("RMS Power")
    plt.title("RMS Power Spectrum")
    plt.xlim(10, 100)
    plt.ylim(0, 1 * 10 ** (-11))
    plt.grid(True)
    plt.savefig(file, dpi=200, bbox_inches="tight")
    plt.close()
    return freqs, rms_spectrum, psds


def finding_shift_third(protocol):
    time = len(protocol[1]) / 96000
    data_sample = np.interp(np.arange(0, len(protocol[1]), (len(protocol[1]) / time) / fsamp),
                            np.arange(0, len(protocol[1])), protocol[1])

    # correlate stimuli channel with stimuli protocol to find the time shift and thus start of the clicks
    corr = correlate(proband_stim['65'], data_sample, "valid")
    plt.figure()
    plt.plot(corr)
    plt.plot()

    sorted_indices = np.argsort(np.abs(corr))[::-1]

    # First: index of the highest value
    first_idx = sorted_indices[0]
    selected_indices = []
    # Second: find the next highest index that's at least 100 away
    for idx in sorted_indices:
        if all(abs(idx - sel) > 100 for sel in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) == 6:
            break
    shift_times = []
    for s in selected_indices:
        shift_times.append(proband_stim['time'].iloc[s])

    return np.sort(shift_times)


def finding_shift_noises(protocol):
    time = len(protocol[1]) / 96000
    data_sample = np.interp(np.arange(0, len(protocol[1]), (len(protocol[1]) / time) / fsamp),
                            np.arange(0, len(protocol[1])), protocol[1])

    # correlate stimuli channel with stimuli protocol to find the time shift and thus start of the clicks
    corr = correlate(proband_stim['65'], data_sample, "valid")
    # the min value of the correlation depicts the shift index
    sorted_indices = np.argsort(np.abs(corr))[::-1]

    # First: index of the highest value
    first_idx = sorted_indices[0]

    # Second: find the next highest index that's at least 100 away
    for idx in sorted_indices[1:]:
        if abs(idx - first_idx) > 10000:
            second_idx = idx
            break
    else:
        second_idx = None

    shifts = [first_idx, second_idx]
    shifts = np.sort(shifts)
    shift_time1 = proband_stim['time'].iloc[shifts[0]]
    shift_time2 = proband_stim['time'].iloc[shifts[1]]

    # plot correlation for verification
    # plt.figure(figsize=(10, 3))
    # plt.subplot(2, 2, 1)
    # x = np.arange(len(corr)) / fsamp
    # plt.plot(x, corr)
    # plt.xlabel("Shift [s]")
    # plt.ylabel("Correlation")
    # # plot the match
    # plt.subplot(2, 2, 2)
    # data_shift = [None] * len(proband_stim['65'])
    # data_shift[shifts[0]:shifts[0] + len(data_sample)] = data_sample / 3
    # x = np.arange(len(proband_stim['65'])) / fsamp
    # plt.plot(x, proband_stim['65'], label="Stimuli Channel")
    # plt.plot(x, data_shift, label="Stimuli Protocol")
    # plt.xlim(shift_time1 - 3, shift_time1 + 40)
    # plt.xlabel("Time [s]")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # x = np.arange(len(corr)) / fsamp
    # plt.plot(x, corr)
    # plt.xlabel("Shift [s]")
    # plt.ylabel("Correlation")
    # # plot the match
    # plt.subplot(2, 2, 4)
    # data_shift = [None] * len(proband_stim['65'])
    # data_shift[shifts[1]:shifts[1] + len(data_sample)] = data_sample / 3
    # x = np.arange(len(proband_stim['65'])) / fsamp
    # plt.plot(x, proband_stim['65'], label="Stimuli Channel")
    # plt.plot(x, data_shift, label="Stimuli Protocol")
    # plt.xlim(shift_time2 - 3, shift_time2 + 40)
    # plt.xlabel("Time [s]")
    # plt.legend()
    # plt.show()

    return shift_time1, shift_time2


# first stimuli
evoked_1_with_Notch_2378Hz = []
evoked_1_without_Notch_2378Hz = []
evoked_1_with_Notch_2828Hz = []
evoked_1_without_Notch_2828Hz = []
evoked_1_with_Notch_3364Hz = []
evoked_1_without_Notch_3364Hz = []
# second stimuli
am_with_notch = []
am_without_notch = []
freqs = []
psds_w_n = []
psds_wo_n = []
# third stimuli
notch_with_pause = []
white_with_pause = []
notch_with_pause_onset = []
white_with_pause_onset = []
# fourth stimuli
notch_with_pauses_start = []
notch_with_pauses_stop = []
white_with_pauses_start = []
white_with_pauses_stop = []
diff_onset = []
diff_offset = []

# subjects grouped by classes (ZT = zwickert tone perceived, T = Tinnitus subejcts, bN = bigger Notch)
subjects_ZT = [1, 4, 10, 11, 16]
subjects_T = [13, 22, 28, 30, 31, 32, 33, 34, 36, 37, 39]
subjects = [2, 3, 5, 6, 7, 8, 9, 12, 14, 15, 17, 25, 26, 27, 29, 35, 38]
subjects_bN = [18, 19, 20, 23, 24]
# zt = "T_"
# zt = "All_"
# zt = "ZT_"
zt = ""
# zt = "BiggerNotch_"

for i in range(1, 40):
    if i == 21:
        continue

    if i < 10:
        file = "F:/Zwicker_Study/Messdateien/Zwickerton_ZT_0" + str(i) + ".vhdr"
    else:
        file = "F:/Zwicker_Study/Messdateien/Zwickerton_ZT_" + str(i) + ".vhdr"
    data = mne.io.read_raw_brainvision(file, preload=True)

    file = ".../Prob" + str(i) + "_EEGdata_raw.fif"
    prep_data = mne.io.read_raw_fif(file, preload=True)

    file = ".../Prob" + str(i) + "_stimuli_channel_raw.fif"
    stimuli_channel = mne.io.read_raw_fif(file, preload=True)
    proband_stim = stimuli_channel.to_data_frame()
    fsamp = stimuli_channel.info["sfreq"]

    ############first stimuli

    protocol = np.load(".../First_Stimulus_Tones_without_Notch.npy")
    durations = np.load(".../First_Stimulus_poisson_durations.npy")
    types = np.load(".../First_Stimulus_poisson_types.npy")

    shift_time1, shift_time2 = finding_shift(protocol, proband_stim)
    print("Time Shifts: ", shift_time1, shift_time2)

    shifts_w_notch = [0.05 + shift_time1 + np.sum(durations[:i]) for i in range(len(durations))]
    shifts_wo_notch = [0.05 + shift_time2 + np.sum(durations[:i]) for i in range(len(durations))]

    cov_data = prep_data.crop(tmin=shift_time1-10, tmax=shift_time1, include_tmax=False)
    cov_data.save(".../Cov_Data/Cov_Prob_" + str(i) + "_raw.fif", overwrite=True)


    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_w_notch = np.zeros((len(shifts_w_notch), 3), dtype=int)
    events_w_notch[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in shifts_w_notch]
    events_w_notch[:, 2] = types

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_wo_notch = np.zeros((len(shifts_wo_notch), 3), dtype=int)
    events_wo_notch[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in shifts_wo_notch]
    events_wo_notch[:, 2] = types

    event_code = {"2378Hz": 0, "2828Hz": 1, "3364Hz": 2}

    epochs_w_notch = mne.Epochs(prep_data, events_w_notch, event_id=event_code, tmin=-0.2, tmax=0.8, picks="eeg",
                                preload=True,
                                reject=dict(eeg=0.0001))

    epochs_wo_notch = mne.Epochs(prep_data, events_wo_notch, event_id=event_code, tmin=-0.2, tmax=0.8, picks="eeg",
                                 preload=True,
                                 reject=dict(eeg=0.0001))

    evoked_1_with_Notch_2378Hz.append(epochs_w_notch["2378Hz"].average())
    evoked_1_with_Notch_2828Hz.append(epochs_w_notch["2828Hz"].average())
    evoked_1_with_Notch_3364Hz.append(epochs_w_notch["3364Hz"].average())
    evoked_1_without_Notch_2378Hz.append(epochs_wo_notch["2378Hz"].average())
    evoked_1_without_Notch_2828Hz.append(epochs_wo_notch["2828Hz"].average())
    evoked_1_without_Notch_3364Hz.append(epochs_wo_notch["3364Hz"].average())

    epochs_w_notch["2378Hz"].average().save(".../1_Prob_" + str(i) + "_2378_notch_ave.fif", overwrite=True)
    epochs_w_notch["2828Hz"].average().save(".../1_Prob_" + str(i) + "_2828_notch_ave.fif", overwrite=True)
    epochs_w_notch["3364Hz"].average().save(".../1_Prob_" + str(i) + "_3364_notch_ave.fif", overwrite=True)
    epochs_wo_notch["2378Hz"].average().save(".../1_Prob_" + str(i) + "_2378_white_ave.fif", overwrite=True)
    epochs_wo_notch["2828Hz"].average().save(".../1_Prob_" + str(i) + "_2828_white_ave.fif", overwrite=True)
    epochs_wo_notch["3364Hz"].average().save(".../1_Prob_" + str(i) + "_3364_white_ave.fif", overwrite=True)


    with mne.viz.use_browser_backend("matplotlib"):
        epochs_w_notch["2378Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_2378Hz_with_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_wo_notch["2378Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_2378Hz_without_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_w_notch["2828Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_2828Hz_with_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_wo_notch["2828Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_2828Hz_without_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_w_notch["3364Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_3364Hz_with_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_wo_notch["3364Hz"].average().plot(show=False, ylim=dict(eeg=[-3, 3]))
        file = ".../First/" + zt + "Prob_" + str(i) + "_3364Hz_without_Notch.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()

    ############second stimuli
    protocol = np.load(".../Second_Stimulus_AM_40_with_Notch.npy")

    shift_time1, shift_time2 = finding_shift(protocol, proband_stim)
    print("Time Shifts: ", shift_time1, shift_time2)

    am_40_w_notch = data.copy().pick(picks="eeg").crop(tmin=shift_time1, tmax=shift_time1 + 120)
    am_40_wo_notch = data.copy().pick(picks="eeg").crop(tmin=shift_time2, tmax=shift_time2 + 120)

    file = ".../Second/" + zt + "Prob_" + str(i) + "_AM_40_with_Notch.png"
    freqs, am_result_w_notch, psds_w = get_fft(am_40_w_notch, file)
    file = ".../Second/" + zt + "Prob_" + str(i) + "_AM_40_without_Notch.png"
    freqs, am_result_wo_notch, psds_wo = get_fft(am_40_wo_notch, file)
    am_with_notch.append(am_result_w_notch)
    am_without_notch.append(am_result_wo_notch)
    psds_w_n.append(psds_w)
    psds_wo_n.append(psds_wo)

    ############third stimuli
    protocol = np.load("F:/Zwicker_Study/Stimuli_Files/Third_Stimulus_Notch_with_Pause.npy")
    shifts = finding_shift_third(protocol)
    print("Shifts: ", shifts)
    pause_w_notch = [shifts[0] + 40, shifts[1] + 40, shifts[2] + 40]
    pause_w_white = [shifts[3] + 40, shifts[4] + 40, shifts[5] + 40]
    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pause_w_notch = np.zeros((3, 3), dtype=int)
    events_pause_w_notch[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pause_w_notch]
    events_pause_w_notch[:, 2] = np.ones(len(pause_w_notch))

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pause_w_white = np.zeros((3, 3), dtype=int)
    events_pause_w_white[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pause_w_white]
    events_pause_w_white[:, 2] = np.ones(len(pause_w_white))

    epochs_pause_w_notch = mne.Epochs(prep_data, events_pause_w_notch, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                      reject=dict(eeg=0.0001))

    epochs_pause_w_white = mne.Epochs(prep_data, events_pause_w_white, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                      reject=dict(eeg=0.0001))

    notch_with_pause.append(epochs_pause_w_notch.average())
    white_with_pause.append(epochs_pause_w_white.average())

    with mne.viz.use_browser_backend("matplotlib"):
        epochs_pause_w_notch.average().plot(show=False)#, ylim=dict(eeg=[-3, 3]))
        file = ".../Third/" + zt + "Prob_" + str(i) + "_Notch_with_pause_offset.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_pause_w_white.average().plot(show=False)#, ylim=dict(eeg=[-3, 3]))
        file = ".../Third/" + zt + "Prob_" + str(i) + "_White_with_pause_offset.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()

    epochs_pause_w_notch.average().save(".../3_Prob" + str(i) + "_notch_ave.fif", overwrite=True)
    epochs_pause_w_white.average().save(".../3_Prob" + str(i) + "_white_ave.fif", overwrite=True)

    epochs_pause_w_notch.save(".../3_Prob" + str(i) + "_notch_epo.fif",
                                        overwrite=True)
    epochs_pause_w_white.save(".../3_Prob" + str(i) + "_white_epo.fif",
                                        overwrite=True)

    pause_w_notch_onset = [shifts[0], shifts[1], shifts[2]]
    pause_w_white_onset = [shifts[3], shifts[4], shifts[5]]
    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pause_w_notch_onset = np.zeros((3, 3), dtype=int)
    events_pause_w_notch_onset[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pause_w_notch_onset]
    events_pause_w_notch_onset[:, 2] = np.ones(len(pause_w_notch_onset))

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pause_w_white_onset = np.zeros((3, 3), dtype=int)
    events_pause_w_white_onset[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pause_w_white_onset]
    events_pause_w_white_onset[:, 2] = np.ones(len(pause_w_white_onset))

    epochs_pause_w_notch_onset = mne.Epochs(prep_data, events_pause_w_notch_onset, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                      reject=dict(eeg=0.0001))

    epochs_pause_w_white_onset = mne.Epochs(prep_data, events_pause_w_white_onset, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                      reject=dict(eeg=0.0001))

    notch_with_pause_onset.append(epochs_pause_w_notch_onset.average())
    white_with_pause_onset.append(epochs_pause_w_white_onset.average())

    with mne.viz.use_browser_backend("matplotlib"):
        epochs_pause_w_notch_onset.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Third/" + zt + "Prob_" + str(i) + "_Notch_with_pause_onset.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_pause_w_white_onset.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Third/" + zt + "Prob_" + str(i) + "_White_with_pause_onset.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()


    # ############fourth stimuli
    protocol = np.load(".../Fourth_Stimulus_Notch_with_Pauses.npy")
    shift_time1, shift_time2 = finding_shift_noises(protocol)
    print("Time Shifts: ", shift_time1, shift_time2)
    durations = np.load(".../Fourth_Stimulus_poisson_durations.npy")
    pauses_w_notch = [shift_time1 + np.sum(durations[:i]) for i in range(len(durations))]
    pauses_w_white = [shift_time2 + np.sum(durations[:i]) for i in range(len(durations))]

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pauses_w_notch = np.zeros((len(pauses_w_notch), 3), dtype=int)
    events_pauses_w_notch[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pauses_w_notch]
    events_pauses_w_notch[:, 2] = np.ones(len(pauses_w_notch))

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pauses_w_white = np.zeros((len(pauses_w_white), 3), dtype=int)
    events_pauses_w_white[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pauses_w_white]
    events_pauses_w_white[:, 2] = np.ones(len(pauses_w_white))

    epochs_pauses_w_notch = mne.Epochs(prep_data, events_pauses_w_notch, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                       reject=dict(eeg=0.0001))

    epochs_pauses_w_white = mne.Epochs(prep_data, events_pauses_w_white, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                       reject=dict(eeg=0.0001))

    notch_with_pauses_start.append(epochs_pauses_w_notch.average())
    white_with_pauses_start.append(epochs_pauses_w_white.average())

    epochs_pauses_w_notch.average().save(".../4_Prob" + str(i) + "_notch_onset_ave.fif", overwrite=True)
    epochs_pauses_w_white.average().save(".../4_Prob" + str(i) + "_white_onset_ave.fif", overwrite=True)
    #
    epochs_pauses_w_notch.save(
        ".../4_Prob" + str(i) + "_notch_onset_epo.fif", overwrite=True)
    epochs_pauses_w_white.save(
        ".../4_Prob" + str(i) + "_white_onset_epo.fif", overwrite=True)

    with mne.viz.use_browser_backend("matplotlib"):
        epochs_pauses_w_notch.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Fourth/" + zt + "Prob_" + str(i) + "_Notch_start.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_pauses_w_white.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Fourth/" + zt + "Prob_" + str(i) + "_White_start.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()

    pauses_notch = [shift_time1 + 2 + np.sum(durations[:i]) for i in range(len(durations))]
    pauses_white = [shift_time2 + 2 + np.sum(durations[:i]) for i in range(len(durations))]

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pauses_notch = np.zeros((len(pauses_notch), 3), dtype=int)
    events_pauses_notch[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pauses_notch]
    events_pauses_notch[:, 2] = np.ones(len(pauses_notch))

    _, time_eeg = prep_data.get_data(picks="eeg", return_times=True)
    events_pauses_white = np.zeros((len(pauses_white), 3), dtype=int)
    events_pauses_white[:, 0] = [np.argmin(np.abs(np.array(time_eeg) - i)) for i in pauses_white]
    events_pauses_white[:, 2] = np.ones(len(pauses_white))

    epochs_pauses_notch = mne.Epochs(prep_data, events_pauses_notch, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                     reject=dict(eeg=0.0001))

    epochs_pauses_white = mne.Epochs(prep_data, events_pauses_white, tmin=-0.5, tmax=2, picks="eeg", preload=True,
                                     reject=dict(eeg=0.0001))

    notch_with_pauses_stop.append(epochs_pauses_notch.average())
    white_with_pauses_stop.append(epochs_pauses_white.average())

    epochs_pauses_notch.average().save(".../4_Prob" + str(i) + "_notch_offset_ave.fif", overwrite=True)
    epochs_pauses_white.average().save(".../4_Prob" + str(i) + "_white_offset_ave.fif", overwrite=True)

    epochs_pauses_notch.save(
        ".../4_Prob" + str(i) + "_notch_offset_epo.fif", overwrite=True)
    epochs_pauses_white.save(
        ".../4_Prob" + str(i) + "_white_offset_epo.fif", overwrite=True)

    with mne.viz.use_browser_backend("matplotlib"):
        epochs_pauses_notch.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Fourth/" + zt + "Prob_" + str(i) + "_Notch_stop.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()
        epochs_pauses_white.average().plot(show=False)  # , ylim=dict(eeg=[-3, 3]))
        file = ".../Fourth/" + zt + "Prob_" + str(i) + "_White_stop.png"
        plt.savefig(file, dpi=200, bbox_inches="tight")
        plt.close()


# first AVG
evoked_1_avg_with_Notch_2378Hz = mne.combine_evoked(evoked_1_with_Notch_2378Hz, weights="nave")
evoked_1_avg_without_Notch_2378Hz = mne.combine_evoked(evoked_1_without_Notch_2378Hz, weights="nave")
evoked_1_avg_with_Notch_2828Hz = mne.combine_evoked(evoked_1_with_Notch_2828Hz, weights="nave")
evoked_1_avg_without_Notch_2828Hz = mne.combine_evoked(evoked_1_without_Notch_2828Hz, weights="nave")
evoked_1_avg_with_Notch_3364Hz = mne.combine_evoked(evoked_1_with_Notch_3364Hz, weights="nave")
evoked_1_avg_without_Notch_3364Hz = mne.combine_evoked(evoked_1_without_Notch_3364Hz, weights="nave")

evoked_1_avg_with_Notch_2378Hz.save(".../1_2378_notch_ave.fif", overwrite=True)
evoked_1_avg_without_Notch_2378Hz.save(".../1_2378_white_ave.fif", overwrite=True)
evoked_1_avg_with_Notch_2828Hz.save(".../1_2828_notch_ave.fif", overwrite=True)
evoked_1_avg_without_Notch_2828Hz.save(".../1_2828_white_ave.fif", overwrite=True)
evoked_1_avg_with_Notch_3364Hz.save(".../1_3364_notch_ave.fif", overwrite=True)
evoked_1_avg_without_Notch_3364Hz.save(".../1_3364_white_ave.fif", overwrite=True)

# second AVG
np.save("AM_with_notch.npy", np.array(am_with_notch))
np.save("AM_without_notch.npy", np.array(am_without_notch))
np.save("freqs.npy", freqs)
np.save("psds_w_n.npy", psds_w_n)
np.save("psds_w_no.npy", psds_wo_n)

plt.figure()
plt.vlines(40, 0, 1, linestyle="dashed", color="orange")
plt.plot(freqs, np.mean(am_with_notch, axis=0), linewidth=3)
plt.xlabel("Frequency (Hz)")
plt.ylabel("RMS Power")
plt.title("RMS Power Spectrum")
plt.xlim(20, 60)
plt.ylim(0, 1 * 10 ** (-11))
plt.grid(True)
file = ".../2_" + zt + "AVG_AM_with_notch.png"
plt.savefig(file, dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.vlines(40, 0, 1, linestyle="dashed", color="orange")
min_ = 10000000
for i in am_without_notch:
    if i.shape[0] < min_:
        min_ = i.shape[0]
am_without_notch_ = []
for i in range(len(am_without_notch)):
    am_without_notch_.append(am_without_notch[i][:min_])

plt.plot(freqs, np.mean(am_without_notch_, axis=0), linewidth=3)
plt.xlabel("Frequency (Hz)")
plt.ylabel("RMS Power")
plt.title("RMS Power Spectrum")
plt.xlim(20, 60)
plt.ylim(0, 1 * 10 ** (-11))
plt.grid(True)
file = ".../2_" + zt + "AVG_AM_without_notch.png"
plt.savefig(file, dpi=200, bbox_inches="tight")
plt.close()

# third AVG
notch_with_pause = mne.combine_evoked(notch_with_pause, weights="nave")
white_with_pause = mne.combine_evoked(white_with_pause, weights="nave")

notch_with_pause.filter(1, 8)
white_with_pause.filter(1, 8)

with mne.viz.use_browser_backend("matplotlib"):
    notch_with_pause.plot(show=False, ylim=dict(eeg=[-5, 5]))
    file = ".../3_" + zt + "AVG_Notch_with_Pause_offset.png"
    plt.savefig(file, dpi=200, bbox_inches="tight")
    plt.close()
    white_with_pause.plot(show=False, ylim=dict(eeg=[-5, 5]))
    file = ".../3_" + zt + "AVG_White_with_Pause_offset.png"
    plt.savefig(file, dpi=200, bbox_inches="tight")
    plt.close()

notch_with_pause_onset = mne.combine_evoked(notch_with_pause_onset, weights="nave")
white_with_pause_onset = mne.combine_evoked(white_with_pause_onset, weights="nave")

notch_with_pause_onset.filter(1, 8)
white_with_pause_onset.filter(1, 8)

with mne.viz.use_browser_backend("matplotlib"):
    notch_with_pause_onset.plot(show=False, ylim=dict(eeg=[-5, 5]))
    file = ".../3_" + zt + "AVG_Notch_with_Pause_onset.png"
    plt.savefig(file, dpi=200, bbox_inches="tight")
    plt.close()
    white_with_pause_onset.plot(show=False, ylim=dict(eeg=[-5, 5]))
    file = ".../3_" + zt + "AVG_White_with_Pause_onset.png"
    plt.savefig(file, dpi=200, bbox_inches="tight")
    plt.close()

# fourth AVG
notch_with_pauses_start = mne.combine_evoked(notch_with_pauses_start, weights="nave")
notch_with_pauses_stop = mne.combine_evoked(notch_with_pauses_stop, weights="nave")
white_with_pauses_start = mne.combine_evoked(white_with_pauses_start, weights="nave")
white_with_pauses_stop = mne.combine_evoked(white_with_pauses_stop, weights="nave")

