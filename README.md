# Zwickertone EEG Study
## Mechanisms of Zwicker Tone Induction: Notched Noise Changes Neural Processing without Evidence for Central Gain or Auditory Enhancement

This repository explains the processing and methods used to analyze the EEG. The preprocessed EEG data with the according stimuli channels and the stimuli files can be downloaded from Zenodo (https://zenodo.org/records/18186380) "Zwickerton EEG Study".

In this study we used 4 different stimuli with corresponding control stimuli: \
First: three pure tones during - and without notch noise \
Second: AM pure tone during - and without notch noise \
Third: 40s notch noise with 10s pause x 3 times - 40s white noise with 10s pause x 3 times \
Fourth: for two minutes each 2s notch noises with 1s pauses - 2s white noises with 1s pauses

The python scripts are used for evaluating the EEG data like following:

1. Aligning and segmenting data for the specific stimuli: "Zwickerton_Evaluation_github.py"
2. Evaluating data for first stimulus: "1_evaluation_github.py"
3. Evaluating data for second stimulus: "2_evaluating_github.py"
4. Comparing different models using data from third stimulus: "3_classification_model_comparison_github.py"
5. Plotting clusters using data from third stimulus and calculating GDV values: "3_cluster_gdv_github.py"
6. Frequency analysis of offset responses using dat of fourth stimulus: "4_frequency_analysis_github.py"
7. Statistically evaluating ERPs offset responses: "4_offset_evaluation_github.py"
