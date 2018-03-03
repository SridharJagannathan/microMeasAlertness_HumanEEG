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
S.model_filename = ['model_collec64_'];%old_model_collec64_
modelfilepath = [ S.model_filepath S.model_filename];

addpath(ftp_toolbox);
addpath(genpath(eeglab_toolbox));
addpath(genpath(uicromeasures_toolbox));

%% %1. Preprocessed file -- > This contains the EEGlab preprocessed file

subject_ids = {'5','7','9','13','15', ...
               '17','19','20','21','23','24', ...
               '26','27','28','29','30','36', ...
               '41','42','44','45','48','49', ...
               '51','53','54','55','56','58', ...
               '60'};
           
for m = 1 : length(subject_ids)           

testsubj = subject_ids{m};
testsubj = str2num(testsubj);

S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
S.eeg_filename = ['AuMa_' num2str(testsubj) '_pretrial_preprocess'];

% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Use it to micro measure alertness levels..
[trialstruc] = classify_microMeasures(EEG, modelfilepath,'128');

%% Now validate that with Hori..
%2. Horiscale data  --> Common for all subjects
S.hori_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess/'];
S.hori_filename = 'merged_AudMaskingCz_hori.mat';

hori_data = load([S.hori_filepath S.hori_filename]);
Goldscore = hori_data.horidataset.HoridataV;
allsubj_id = hori_data.horidataset.subj_id;

testtrls = find(allsubj_id == testsubj);
Goldusable = Goldscore(testtrls);

gold_test = nan(length(Goldusable),1);

alert_trls = find(Goldusable<=2);
ripple_trls = intersect(find(Goldusable>=3),find(Goldusable<=5));
grapho_trls = find(Goldusable>=6);

gold_test(alert_trls) = 1;
gold_test(ripple_trls) = 2;
gold_test(grapho_trls) = 3;

algo_test = nan(EEG.trials,1);
algo_test(trialstruc.alert) = 1;
algo_test(trialstruc.drowsymild) = 2;
algo_test(trialstruc.drowsysevere) = 3;

[confusionMatrixAll,orderAll] = confusionmat(gold_test,algo_test);
% Calculate the overall accuracy from the overall predicted class label
accuracyAll = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));

fprintf('\n--Validating :%s--\n',string(testsubj));

fprintf('-- Accuracy rate %0.2f%% --- \n', 100*accuracyAll);

if length(confusionMatrixAll) == 1
    if unique(testLabel) == 1
        names= {'Alert'}; 
    else
        names= {'Drowsy'}; 
    end
elseif length(confusionMatrixAll) == 2
       if orderAll(1) == 1
           names= {'Alert','Ripples'};
       else
           names= {'Ripples','Grapho'};
       end
elseif length(confusionMatrixAll) == 3
    names= {'Alert','Ripples','Grapho'};
elseif length(confusionMatrixAll) == 4
    names= {'Alert','Ripples','Vertex','Spindle'};
 end
disptable = array2table( confusionMatrixAll, 'VariableNames', names, 'RowNames', names );
disp(disptable);

end

temp =[];






