clear
%clc
%close all

pathappend = '/work/imagingQ/';

%% Add paths now..

%Fieldtrip path
ftp_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/fieldtrip-20151223'];
%EEGlab path
eeglab_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/eeglab13_5_4b'];
%uicromeasures path
uicromeasures_toolbox = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/microMeasAlertness_HumanEEG'];
%Model file -- > This contains the Model file for classification
S.model_filepath = [ uicromeasures_toolbox '/models/'];
S.model_filename = ['model_collec64_'];
modelfilepath = [ S.model_filepath S.model_filename];

addpath(ftp_toolbox);
addpath(genpath(eeglab_toolbox));
addpath(genpath(uicromeasures_toolbox));
rmpath(genpath([eeglab_toolbox '/functions/octavefunc'])); 

%% %1. Preprocessed file -- > This contains the EEGlab preprocessed file

%S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
%S.eeg_filename = ['AuMa_5_pretrial_preprocess'];

S.eeg_filepath = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/pretrial_preprocess'];
S.eeg_filename = ['105_pretrial_preprocess'];

% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Use it to micro measure alertness levels..
% Outputs: trialstruc   - Trial indexs of alert, drowsy(mild),
%                         drowsy(severe), and also vertex, spindle, 
%                         k-complex indices..
[trialstruc] = classify_microMeasures(EEG, modelfilepath);