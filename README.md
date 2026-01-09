# EEG Study
## Mechanisms of Zwicker Tone Induction: Notched Noise Changes Neural Processing without Evidence for Central Gain or Auditory Enhancement

This repository explains the processing and methods used to analyze the EEG. The preprocessed EEG data with the according stimuli channels and the stimuli files can be downloaded from Zenodo (https://zenodo.org/records/18186380) "Zwickerton EEG Study".

In this study we used 4 different stimuli with corresponding control stimuli: 
  First: three pure tones during - and without notch noise
  Second: AM pure tone during - and without notch noise
  Third: 40s notch noise with 10s pause x 3 times - 40s white noise with 10s pause x 3 times
  Fourth: for two minutes each 2s notch noises with 1s pauses - 2s white noises with 1s pauses

The python scripts are used for evaluating the EEG data like following:

1. 
2. MEG data was preprocessed using MEG_preprocess_OpenAccess.py
3. For aligning the audio book with stimuli channels we used Forced Alignment https://clarin.phonetik.uni-muenchen.de/BASWebServices/interface/WebMAUSBasic with the audio signals and transcript to get the word onsets
4. Each word in the transcript was classified into word classes using spaCy https://spacy.io/
