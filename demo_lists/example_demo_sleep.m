clear
%clc
%close all

%% Add paths now..
% Set these to your local installations..
%Fieldtrip path
ftp_toolbox = '/path/to/fieldtrip-20151223';
%EEGlab path
eeglab_toolbox = '/path/to/eeglab13_5_4b';

%uicromeasures path (parent folder of demo_lists)
uicromeasures_toolbox = fileparts(fileparts(mfilename('fullpath')));
%Model file -- > This contains the Model file for classification
S.model_filepath = [ uicromeasures_toolbox '/models/'];
S.model_filename = ['model_collec_elecsleep'];
modelfilepath = [ S.model_filepath S.model_filename];

addpath(ftp_toolbox);
addpath(genpath(eeglab_toolbox));
addpath(genpath(uicromeasures_toolbox));
rmpath(genpath([eeglab_toolbox '/functions/octavefunc']));

%% %1. Preprocessed file -- > This contains the EEGlab preprocessed file
% Set these to your sleep montage recording..
S.eeg_filepath = '/path/to/your/data/';
S.eeg_filename = 'your_sleep_montage_recording';

% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Some preprocessing for montage..
% rename electrodes according to the 10-20 system, edit the mapping below
% to match the channel labels of your recording system..
stdnamestruct = struct('EEG004','F3', 'EEG005','Fz', 'EEG006','F4', ...
                       'EEG009','C3', 'EEG011','C4', ...
                       'EEG014','P3', 'EEG015','Pz', 'EEG016','P4', ...
                       'EEG018','O1', 'EEG019','O2', ...
                       'EEG020','A1', 'EEG021','A2', 'EEG022','HREOG');

for idx = 1:length(EEG.chanlocs)
    tmplabel = EEG.chanlocs(idx).labels;
    if isfield(stdnamestruct, tmplabel)
        EEG.chanlocs(idx).labels = stdnamestruct.(tmplabel);
    end
end

%% Use it to micro measure alertness levels..
% Outputs: trialstruc   - Trial indexs of alert, drowsy(mild),
%                         drowsy(severe), and also vertex, spindle,
%                         k-complex indices..
channelconfig = 'sleep';  %'64' or '128' or '256' or 'sleep' channel eeg configuration..
[trialstruc] = classify_microMeasures(EEG, modelfilepath,channelconfig);
