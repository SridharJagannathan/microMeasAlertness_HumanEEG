clear
%clc
%close all

%% Add paths now..
% Set these to your local installations..
%Fieldtrip path
ftp_toolbox = '/path/to/fieldtrip-20151223';
%EEGlab path
eeglab_toolbox = '/path/to/eeglab13_5_4b';

%uicromeasures path (folder containing this script)
uicromeasures_toolbox = fileparts(mfilename('fullpath'));
%Model file -- > This contains the Model file for classification
S.model_filepath = [ uicromeasures_toolbox '/models/'];
S.model_filename = ['model_collec_elec64'];%'model_collec_elec64','model_collec_elecsleep'
modelfilepath = [ S.model_filepath S.model_filename];

addpath(ftp_toolbox);
addpath(genpath(eeglab_toolbox));
addpath(genpath(uicromeasures_toolbox));
rmpath(genpath([eeglab_toolbox '/functions/octavefunc']));

%% %1. Preprocessed file -- > This contains the EEGlab preprocessed file
% Example dataset bundled with the toolbox (64 channel Neuroscan, subject 122
% from Jagannathan et al., NeuroImage 2018)..
S.eeg_filepath = [ uicromeasures_toolbox '/example_data/'];
S.eeg_filename = ['122_pretrial_preprocess'];

% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Use it to micro measure alertness levels..
% Outputs: trialstruc   - Trial indexs of alert, drowsy(mild),
%                         drowsy(severe), and also vertex, spindle,
%                         k-complex indices..
channelconfig = '64'; %'64' or '128' or '256' or 'sleep' channel eeg configuration..
[trialstruc] = classify_microMeasures(EEG, modelfilepath,channelconfig);

fprintf('\n--Alert: %d, Drowsy(mild): %d, Drowsy(severe): %d of %d trials--\n', ...
        length(trialstruc.alert), length(trialstruc.drowsymild), ...
        length(trialstruc.drowsysevere), EEG.trials);
