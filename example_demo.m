clear
clc
%close all

pathappend = '/rds/project/tb419/rds-tb419-bekinschtein/Sri/';

%% Add paths now..

%Fieldtrip path
ftp_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/fieldtrip-20151223'];
%EEGlab path
eeglab_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/eeglab13_5_4b'];
%uicromeasures path
uicromeasures_toolbox = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/microMeasAlertness_HumanEEG'];
%Model file -- > This contains the Model file for classification
S.model_filepath = [ uicromeasures_toolbox '/models/'];
S.model_filename = ['model_collec_elecsleep'];%'model_collec_elecsleep','model_collec_elec64'
modelfilepath = [ S.model_filepath S.model_filename];

addpath(ftp_toolbox);
addpath(genpath(eeglab_toolbox));
addpath(genpath(uicromeasures_toolbox));
rmpath(genpath([eeglab_toolbox '/functions/octavefunc'])); 

%% %1. Preprocessed file -- > This contains the EEGlab preprocessed file

%S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
%S.eeg_filename = ['AuMa_5_pretrial_preprocess'];

%S.eeg_filepath = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/pretrial_preprocess'];
%S.eeg_filename = ['105_pretrial_preprocess'];

S.eeg_filepath = ['/home/srj34/Downloads/Shri_Files/'];
S.eeg_filename = ['1986100SIBK_resting'];

% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Some preprocessing for montage..
% rename electrodes according to the 10-20 system
origname1 = 'EEG004';  stdname1 = 'F3';
origname2 = 'EEG005';  stdname2 = 'Fz';
origname3 = 'EEG006';  stdname3 = 'F4';
origname4 = 'EEG009';  stdname4 = 'C3';
origname5 = 'EEG011';  stdname5 = 'C4';
origname6 = 'EEG014';  stdname6 = 'P3';
origname7 = 'EEG015';  stdname7 = 'Pz';
origname8 = 'EEG016';  stdname8 = 'P4';
origname9 = 'EEG018';  stdname9 = 'O1';
origname10 = 'EEG019';  stdname10 = 'O2';
origname11 = 'EEG020';  stdname11 = 'A1';
origname12 = 'EEG021';  stdname12 = 'A2';
origname13 = 'EEG022';  stdname13 = 'HREOG';

stdnamestruct = struct(origname1,stdname1,origname2,stdname2,origname3,stdname3,origname4,stdname4,...
                       origname5,stdname5,origname6,stdname6,origname7,stdname7,origname8,stdname8,...
                       origname9,stdname9,origname10,stdname10,origname11,stdname11,origname12,stdname12,...
                       origname13,stdname13);


for idx = 1:length(EEG.chanlocs)
    tmplabel = EEG.chanlocs(idx).labels;
    EEG.chanlocs(idx).labels = stdnamestruct.(tmplabel);
    
end

%% Use it to micro measure alertness levels..
% Outputs: trialstruc   - Trial indexs of alert, drowsy(mild),
%                         drowsy(severe), and also vertex, spindle, 
%                         k-complex indices..

channelconfig = 'sleep';  %'64' or '128' or '256' or 'sleep' channel eeg configuration..
[trialstruc] = classify_microMeasures(EEG, modelfilepath,channelconfig);

tempval = [];