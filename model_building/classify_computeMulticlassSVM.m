%% Start afresh by clearing all command windows and variables
%clc; 
clear;

rng(1); % For reproducibility

%% Load the paths..
loadpath;

%% Load the input files..
subject_ids = {'merged'};

rmsubject_ids = {'','105','107','109','111','113','117', ...
               '118','122','123','125','127','129', ...%, missing data..,'132','151'
               '134','137','138','139','144','147', ...      
               '149','150'};
           
for k = 1: 1 %length(rmsubject_ids)         
           
for m = 1:length(subject_ids)
    
subject = subject_ids{m};
   
%1. Preprocessed file -- > This contains the EEGlab preprocessed file
S.eeg_filepath = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/pretrial_preprocess'];
S.eeg_filename = [subject '_pretrial_preprocess'];

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);

evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

%load the preprocessed EEGdata set..
[T,EEG] = evalc(evalexp);

%2. Load the class labels..Horiscale data  --> Common for all subjects
S.hori_filepath = [pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/hori_data/'];
S.hori_filename = 'merged_hori.mat';

hori_data = load([S.hori_filepath S.hori_filename]);
Goldscore = hori_data.horidataset.HoridataGold;
allsubj_id = hori_data.horidataset.subj_id;

%3. Choose the usable labels..
nc_trls = find(strcmp(Goldscore,'nc'));
na_trls = find(strcmp(Goldscore,'na'));
%nb_trls = find(strcmp(Goldscore,'Spindle'));
rm_subjs = find(allsubj_id == str2double(rmsubject_ids(k)));

fprintf('\n--Removing from the model: subj_%d--\n',str2double(rmsubject_ids(k)));

 %rm_trls = union(nc_trls,na_trls);
 %rm_trls = union(rm_trls,rm_subjs);
 rm_trls =[];

Gold_usable = Goldscore;
%Gold_usable(rm_trls)=[];

indexes = nan(length(Gold_usable),1);
indexes(find(strcmp(Gold_usable,'Alert'))) = 1;
indexes(find(strcmp(Gold_usable,'Ripples'))) = 2;
indexes(find(strcmp(Gold_usable,'Grapho'))) = 2;



graphoindexes = nan(length(Gold_usable),1);
graphoindexes(find(strcmp(Gold_usable,'Ripples'))) = 2;
graphoindexes(find(strcmp(Gold_usable,'Grapho'))) = 3;


S.feat_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/Scripts/classify_libsvm/'];
S.feat_filename = ['feat_groups.mat'];

if ~exist([S.feat_filepath S.feat_filename], 'file')   %Check if features have been computed


evalexp = 'pop_select(EEG, ''notrial'', rm_trls)';

%load the usable EEGdata set..
[T,EEG] = evalc(evalexp);

%% Use only some channels ..
electrodes_rx = {'Oz','O1','O2',...
                 'C3', 'C4', ...
                 'PO10',...
                 'T7','T8','TP8','FT10','TP10',...
                 'F7', 'F8', 'Fz'};
    
chanlabels={EEG.chanlocs.labels};
selec_elec = ismember(chanlabels,electrodes_rx);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_rx] = evalc(evalexp);

EEG = EEG_rx;


%Collect channel labels..
chanlabels={EEG.chanlocs.labels};
electrodes_occ = {'Oz','O1','O2'};
electrodes_tempero = {'T8','TP8','FT10','TP10'};
electrodes_frontal = {'F7', 'F8', 'Fz'};
electrodes_central = {'C3', 'C4'};
electrodes_parietal = {'PO10'};

selec_elec = ismember(chanlabels,electrodes_occ);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_occ] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_frontal);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_front] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_tempero);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_tempero] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_central);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_centro] = evalc(evalexp);

selec_elec = ismember(chanlabels,electrodes_parietal);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_parieto] = evalc(evalexp);

%% Computing Alert features..

fprintf('\n--Computing Variance features--\n');

[trials_alert, misc_alert]= classify_detectAlertTrials(EEG_occ);

fprintf('\n--Computing Coherence features--\n');
eleclabels.frontal = {'F7', 'F8', 'Fz'};
eleclabels.central = {'C3', 'C4'};
eleclabels.parietal = {'Pz'};
eleclabels.temporal =  {'T7', 'T8'};
eleclabels.occipetal = {'Oz','O1', 'O2'};

[coh]= classify_detectCoherenceTrials(EEG,eleclabels);
coh_features = table2array(coh);
    %coh_features = coh;

%  fprintf('\n--Computing wPLI features--\n');
% eleclabels =[];
% eleclabels.frontal = {'F7', 'F8', 'Fz'};
% eleclabels.occipetal = {'Oz','O1', 'O2'};
%  [wPLI]= classify_detectwPLITrials(EEG,eleclabels);

%% Computing Vertex features..
fprintf('\n--Computing Vertex: monophasic features--\n');
monophasic_fts =[]; Data = double((EEG_parieto.data));
for z = 1:EEG_parieto.trials
    for s = 1:size(Data,1)
        
        [Vertex, Vertex_ft] = classify_computeVertexMonophasicfts(Data(s,:,z), EEG_parieto.srate);
        if (Vertex_ft.count>0 && Vertex_ft.negpks_1 < -15 && Vertex_ft.negpks_2 < -15 &&...
                                Vertex_ft.duration >0.1 && Vertex_ft.pospks> 30)
            monophasic_fts(s,z) = 1;
        else
            monophasic_fts(s,z) = 0;

        end
    end
end

fprintf('\n--Computing Vertex: biphasic features--\n');
biphasic_fts =[]; Data = double((EEG_parieto.data));
for z = 1:EEG_parieto.trials
    for s = 1:size(Data,1)
        
        [Vertex, Vertex_ft] = classify_computeVertexBiphasicfts(Data(s,:,z), EEG_parieto.srate);
       if (Vertex_ft.count>0 && Vertex_ft.negpks < -40 && Vertex_ft.pospks> 40) %~isempty(Kcomp.start_stop)
            biphasic_fts(s,z) = 1;
        else
            biphasic_fts(s,z) = 0;

        end
    end
end

monophasic_fts = sum(monophasic_fts,1);
monophasic_def = find(monophasic_fts); 

biphasic_fts = sum(biphasic_fts,1);
biphasic_def = find(biphasic_fts); 


%% Computing spindle features..
fprintf('\n--Computing Spindle: features--\n');

spin_ft =[]; Data = double(squeeze(EEG_tempero.data));
Freq_range = 12:16;Time_params = [0.4 1.5];

for z = 1:EEG_tempero.trials
    
   for s = 1:size(Data,1)
        
    [Spindle, detected_spindles] = classify_computeSpindlefts(Data(s,:,z), EEG_tempero.srate,Freq_range,Time_params);
        if  ~isempty(Spindle.start_stop)%~isempty(spindles_start_end)
            pospeakmean=[];negpeakmean=[];
            for idx = 1:length(detected_spindles)
                pospeakmean(idx) =  detected_spindles{1,idx}.meanpospks;
                
                negpeakmean(idx) =  detected_spindles{1,idx}.meannegpks;
            end
            
             
             if(sum(pospeakmean>9) || sum(negpeakmean>9))
                
               spin_ft(s,z) = 1;
            
             else
                 
               spin_ft(s,z) = nan;
                 
             end 
            
            
        else
            spin_ft(s,z) = 0;

        end
    end
        
end

spinsum_ft = sum(spin_ft,1);
spinnansum_ft = nansum(spin_ft,1);
spinnan_ft = sum(isnan(spin_ft), 1);
spinrecon_cand = intersect(find(spinnansum_ft>=spinnan_ft),find(spinnan_ft>=1));

spinsum_ft = sum(spin_ft,1);
spindle_def = find(spinsum_ft>0);
spindle_def = union(spindle_def,spinrecon_cand);

%% Computing k-complex features..

fprintf('\n--Computing K-complex: features--\n');

kcomp_ft =[]; Data = double(squeeze(EEG.data));

for z = 1:EEG.trials
    
    [Kcomp, Kcomp_ft] = classify_computeKcomplexfts(Data(s,:,z), EEG.srate);
    
        if (Kcomp_ft.count>0 && Kcomp_ft.negpks < -45 && Kcomp_ft.pospks-Kcomp_ft.negpks > 100 &&...
                Kcomp_ft.pospks > 0.5*abs(Kcomp_ft.negpks)) %~isempty(Kcomp.start_stop)
            kcomp_ft(s,z) = 1;
        else
            kcomp_ft(s,z) = 0;

        end
         
end

kcomp_def = find(sum(kcomp_ft,1));

kcomp_features = zeros(1, EEG.trials);
kcomp_features(kcomp_def)=1;


%% graphical elements
grapho_def = union(monophasic_def,biphasic_def);
grapho_def = union(grapho_def,kcomp_def);
nonspind_def = grapho_def;
grapho_def = union(grapho_def,spindle_def);

grapho_comb = zeros(EEG.trials,1);
grapho_comb(grapho_def) = 1;

spind_comb = zeros(EEG.trials,1);
spind_comb(spindle_def) = 1;

nonspind_comb = zeros(EEG.trials,1);
nonspind_comb(nonspind_def) = 1;

%% Summarize the features..

feat_groups{1,m} = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];
               
save([S.feat_filepath S.feat_filename],'feat_groups','misc_alert','coh_features','grapho_comb','spind_comb','nonspind_comb');

    
else
    load([S.feat_filepath S.feat_filename]);
end


end

%% Arrange the features and labels..
               
feat_groups{1,m} = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];

class_feats = [indexes feat_groups{1,1}];

 rm_trls = union(nc_trls,na_trls);
 rm_trls = union(rm_trls,rm_subjs);
 %rm_trls = union(rm_trls,nb_trls);
 
 class_feats(rm_trls,:) = [];
 grapho_feats = [graphoindexes grapho_comb];
 grapho_feats(rm_trls,:) = [];
 
 spind_feats = [graphoindexes spind_comb];
 spind_feats(~spind_comb,1) = nan;
 spind_feats(rm_trls,:) = [];
 
 nonspind_feats = [graphoindexes nonspind_comb];
 nonspind_feats(~nonspind_comb,1) = nan;
 nonspind_feats(rm_trls,:) = [];
 

DataClass = class_feats(:,1);
DataSet = class_feats(:,2:end);

GraphoDataClass = grapho_feats(:,1);
GraphoDataSet = grapho_feats(:,2:end);

SpindDataClass = spind_feats(:,1);
SpindDataSet = spind_feats(:,2:end);

nonSpindDataClass = nonspind_feats(:,1);
nonSpindDataSet = nonspind_feats(:,2:end);


nfold_cv = 5;
c = cvpartition(DataClass,'KFold',nfold_cv);
trainIdx =[];testIdx =[];

model_collecAlert = [];model_collecGrapho = [];
ranges_collAlert =[]; ranges_collGrapho =[]; 
minimums_collAlert =[];minimums_collGrapho =[];

statstrain_collec=[];
statstest_collec=[];

 for kparts = 1:c.NumTestSets


trainIdx = c.training(kparts);
testIdx = c.test(kparts);

trainData = DataSet(find(trainIdx),:);
trainLabel = DataClass(find(trainIdx));

testData = DataSet(find(testIdx),:);
testLabel = DataClass(find(testIdx));



%To scale the training and testing data in the same way..
minimums = min(trainData, [], 1);
ranges = max(trainData, [], 1) - minimums;

trainData = (trainData - repmat(minimums, size(trainData, 1), 1)) ./ repmat(ranges, size(trainData, 1), 1);
testData = (testData - repmat(minimums, size(testData, 1), 1)) ./ repmat(ranges, size(testData, 1), 1);

% Extract important information
labelList = unique(trainLabel);
NClass = length(labelList);
[Ntrain D] = size(trainData);
run = [1:Ntrain]';

% Make the run index for each observation
% Here we will make them into 5 folds
Ncv_classif = 5;

runCVIndex = mod(run,Ncv_classif)+1;


% First we randomly pick some observations from the training set for parameter selection
tmp = randperm(Ntrain);
evalIndex = tmp(1:ceil(Ntrain/2));
evalData = trainData(evalIndex,:);
evalLabel = trainLabel(evalIndex,:);

optionCV.cmin = 2^(-1);%2^(-5)
optionCV.cmax = 2^(25); %2^(25)
optionCV.gammamin = 2^(-1);%2^(-25)
optionCV.gammamax = 2^(25); %2^(5)
optionCV.stepSize = 1;%1
optionCV.bestLog2c = 0;
optionCV.bestLog2g = log2(1/D);
optionCV.epsilon = 0.005;
optionCV.svmCmd = '-q -t 2 -s 0';
Ncv_param = 3; % Ncv-fold cross validation cross validation

% Put the kernel Phi(data)
[bestc, bestg, bestcv] = KernelParameterSelection(evalLabel, evalData, Ncv_param, optionCV);

cmd = [optionCV.svmCmd,' -b 1 -c ',num2str(bestc),' -g ',num2str(bestg)];
% % % % Train the SVM
% % % 
bestModel = svmtrain(trainLabel, trainData, cmd);


% N-cross validation
[predictedLabel, accuracy, decisValueWinner] = svmNFoldCrossValidation(trainLabel, trainData, runCVIndex, cmd);

[confusionMatrixAll,orderAll] = confusionmat(trainLabel,predictedLabel);

%% train the classifier for pruning the spindle elements now..

GraphotrainData = trainData; %GraphoDataSet(find(trainIdx),:);
GraphotrainLabel = GraphoDataClass(find(trainIdx));
GraphotrainDataset = GraphoDataSet(find(trainIdx),:);

GraphotestData = testData; %GraphoDataSet(find(testIdx),:);
GraphotestLabel = GraphoDataClass(find(testIdx));
GraphotestDataset = GraphoDataSet(find(testIdx),:);


SpindtrainData = trainData; %SpindDataSet(find(trainIdx),:);
SpindtrainLabel = SpindDataClass(find(trainIdx));
SpindtrainDataset = SpindDataSet(find(trainIdx),:);

SpindtestData = testData; %SpindDataSet(find(testIdx),:);
SpindtestLabel = SpindDataClass(find(testIdx));
SpindtestDataset = SpindDataSet(find(testIdx),:);

nonSpindtrainData = trainData; %nonSpindDataSet(find(trainIdx),:);
nonSpindtrainLabel = nonSpindDataClass(find(trainIdx));
nonSpindtrainDataset = nonSpindDataSet(find(trainIdx),:);

nonSpindtestData = testData; %nonSpindDataSet(find(testIdx),:);
nonSpindtestLabel = nonSpindDataClass(find(testIdx));
nonSpindtestDataset = nonSpindDataSet(find(testIdx),:);

grapho_pred = intersect(find(predictedLabel==2),find(GraphotrainDataset==1));
grapho_actual= intersect(find(trainLabel==2),find(GraphotrainLabel==3));

spind_pred = intersect(find(predictedLabel==2),find(SpindtrainDataset==1));
spind_actual= intersect(find(trainLabel==2),find(SpindtrainLabel==3));

nonspind_pred = intersect(find(predictedLabel==2),find(nonSpindtrainDataset==1));
nonspind_actual= intersect(find(trainLabel==2),find(nonSpindtrainLabel==3));

traindatagraphoidx = union(spind_pred,spind_actual);
traindatagrapho = SpindtrainData(traindatagraphoidx,:);
trainLabelgrapho = SpindtrainLabel(traindatagraphoidx);

nangraphoidx = find(isnan(trainLabelgrapho));
traindatagrapho(nangraphoidx,:) =[];
trainLabelgrapho(nangraphoidx,:) =[];

%To scale the training and testing data in the same way..
minimumsgrapho = min(traindatagrapho, [], 1);
rangesgrapho = max(traindatagrapho, [], 1) - minimumsgrapho;

traindatagrapho = (traindatagrapho - repmat(minimumsgrapho, size(traindatagrapho, 1), 1)) ./ repmat(rangesgrapho, size(traindatagrapho, 1), 1);


% Extract important information
labelListgrapho = unique(trainLabelgrapho);
NClassgrapho = length(labelListgrapho);
[Ntraingrapho Dgrapho] = size(traindatagrapho);
%if ~exist('run','var')
    rungrapho = [1:Ntraingrapho]';
%end

% Make the run index for each observation
% Here we will make them into 5 folds
Ncv_classifgrapho = 5;

runCVIndexgrapho = mod(rungrapho,Ncv_classifgrapho)+1;


% First we randomly pick some observations from the training set for parameter selection
tmp = randperm(Ntraingrapho);
evalIndex = tmp(1:ceil(Ntraingrapho/2));
evalData = traindatagrapho(evalIndex,:);
evalLabel = trainLabelgrapho(evalIndex,:);

optionCV.cmin = 2^(-1);%2^(-5)
optionCV.cmax = 2^(5); %2^(25)
optionCV.gammamin = 2^(-1);%2^(-25)
optionCV.gammamax = 2^(5); %2^(5)
optionCV.stepSize = 1;%1
optionCV.bestLog2c = 0;
optionCV.bestLog2g = log2(1/D);
optionCV.epsilon = 0.005;
optionCV.svmCmd = '-q -t 2 -s 0';
Ncv_param = 3; % Ncv-fold cross validation cross validation

% Put the kernel Phi(data)
[bestc, bestg, bestcv] = KernelParameterSelection(evalLabel, evalData, Ncv_param, optionCV);

cmd = [optionCV.svmCmd,' -b 1 -c ',num2str(bestc),' -g ',num2str(bestg)];
% % % % Train the SVM
% % % 
bestModelGrapho = svmtrain(trainLabelgrapho, traindatagrapho, cmd);

% N-cross validation
[predictedLabelgrapho, accuracy, decisValueWinner] = svmNFoldCrossValidation(trainLabelgrapho, traindatagrapho, runCVIndexgrapho, cmd);

[confusionMatrixgrapho,ordergrapho] = confusionmat(trainLabelgrapho,predictedLabelgrapho);


grapho_pred = intersect(find(predictedLabel==2),find(GraphotrainDataset==1));
grapho_nonspindpred = intersect(nonspind_pred,grapho_pred);
subtestdatagrapho = GraphotrainData(grapho_pred,:);
subtestLabelgrapho = GraphotrainLabel(grapho_pred);
subtestLabelnonspindgrapho = ismember(grapho_pred,grapho_nonspindpred);

subtestdatagrapho = (subtestdatagrapho - repmat(minimumsgrapho, size(subtestdatagrapho, 1), 1)) ./ repmat(rangesgrapho, size(subtestdatagrapho, 1), 1);


[subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, '-b 1');
%Set the nonspindle element to also grapho..
subpredictedLabelgrapho(subtestLabelnonspindgrapho) = 3;
predactualgrapho = find(subpredictedLabelgrapho ==3);

predictedLabel(grapho_pred(predactualgrapho)) = 3;
trainLabel(grapho_actual) = 3;

[confusionMatrixAll,orderAll] = confusionmat(trainLabel,predictedLabel);


stats_train = confusionmatStats(trainLabel,predictedLabel);

% Calculate the overall accuracy from the overall predicted class label
accuracyAll = trace(confusionMatrixAll)/sum(confusionMatrixAll(:));
disp(['Total train set accuracy is ',num2str(accuracyAll*100),'%']);
fprintf('\n\n');
disp(['########Train set confusion matrix########']);
fprintf('\n\n');
if length(confusionMatrixAll) == 2
   names= {'Alert','Drowsy'};
elseif length(confusionMatrixAll) == 3
    names= {'Alert','Ripples','Grapho'};
elseif length(confusionMatrixAll) == 4
    names= {'Alert','Ripples','Vertex','Spindle'};
 end
disptable = array2table( confusionMatrixAll, 'VariableNames', names, 'RowNames', names );
disp(disptable);

fprintf('\n\n');

% % Use the SVM model to classify the data
  [predictedLabel, accuracy, decisValueWinner] = svmpredict(testLabel, testData, bestModel, '-b 1'); % run the SVM model on the test data
  
  grapho_pred = intersect(find(predictedLabel==2),find(GraphotestDataset==1));
  nonspind_pred = intersect(find(predictedLabel==2),find(nonSpindtestDataset==1));
 
  grapho_nonspindpred = intersect(nonspind_pred,grapho_pred);
  subtestdatagrapho = GraphotestData(grapho_pred,:);
  subtestLabelgrapho = GraphotestLabel(grapho_pred);
  
  subtestLabelnonspindgrapho = ismember(grapho_pred,grapho_nonspindpred);
  
  
  subtestdatagrapho = (subtestdatagrapho - repmat(minimumsgrapho, size(subtestdatagrapho, 1), 1)) ./ repmat(rangesgrapho, size(subtestdatagrapho, 1), 1);

  
  [subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, '-b 1');
  %Set the nonspindle element to also grapho..
  subpredictedLabelgrapho(subtestLabelnonspindgrapho) = 3;
  
  predactualgrapho = find(subpredictedLabelgrapho ==3);
  predictedLabel(grapho_pred(predactualgrapho)) = 3;
  
  grapho_actual= intersect(find(testLabel==2),find(GraphotestLabel==3));
  
  testLabel(grapho_actual) = 3;
  
  [confusionMatrixtest,ordertest] = confusionmat(testLabel,predictedLabel);
  stats_test = confusionmatStats(testLabel,predictedLabel);
 accuracytest = trace(confusionMatrixtest)/sum(confusionMatrixtest(:));
 disp(['Total test set accuracy is ',num2str(accuracytest*100),'%']);

 fprintf('\n\n');
disp(['########Test set confusion matrix########']);
fprintf('\n\n');
if length(confusionMatrixtest) == 2
   names= {'Alert','Drowsy'};
elseif length(confusionMatrixtest) == 3
    names= {'Alert','Ripples','Grapho'};
elseif length(confusionMatrixtest) == 4
    names= {'Alert','Ripples','Vertex','Spindle'};
 end
disptable = array2table( confusionMatrixtest, 'VariableNames', names, 'RowNames', names );
disp(disptable);

fprintf('\n\n');



model_collecAlert = [model_collecAlert bestModel];
ranges_collAlert =[ranges_collAlert; ranges]; 
minimums_collAlert =[minimums_collAlert; minimums];

model_collecGrapho = [model_collecGrapho bestModelGrapho];
ranges_collGrapho =[ranges_collGrapho; rangesgrapho]; 
minimums_collGrapho =[minimums_collGrapho; minimumsgrapho];


statstrain_collec = [statstrain_collec stats_train];
statstest_collec = [statstest_collec stats_test];

end


S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/Scripts/classify_libsvm/'];
S.model_filename = ['model_collec64_' char(rmsubject_ids(k)) ];

save([S.model_filepath S.model_filename],'model_collecAlert','ranges_collAlert','minimums_collAlert',...
                                         'model_collecGrapho','ranges_collGrapho','minimums_collGrapho',...
                                         'statstrain_collec','statstest_collec');
                                     
                                     
S.validation_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/validation/'];
S.validation_filename = ['internal_train_64.mat'];
sensitivity_alert =[];specificity_alert =[];f1_score_alert =[];
sensitivity_ripples =[];specificity_ripples =[];f1_score_ripples =[];
sensitivity_grapho =[];specificity_grapho =[];f1_score_grapho =[];

for idx = 1:length(statstrain_collec)
   
    sensitivity_alert(idx) = 100*statstrain_collec(idx).sensitivity(1);
    sensitivity_ripples(idx) = 100*statstrain_collec(idx).sensitivity(2);
    sensitivity_grapho(idx) = 100*statstrain_collec(idx).sensitivity(3);
    
    specificity_alert(idx) = 100*statstrain_collec(idx).specificity(1);
    specificity_ripples(idx) =100*statstrain_collec(idx).specificity(2);
    specificity_grapho(idx) = 100*statstrain_collec(idx).specificity(3);
    
    f1_score_alert(idx) = statstrain_collec(idx).Fscore(1);
    f1_score_ripples(idx) = statstrain_collec(idx).Fscore(2);
    f1_score_grapho(idx) = statstrain_collec(idx).Fscore(3);
  
end



save([S.validation_filepath S.validation_filename],'sensitivity_alert','sensitivity_ripples','sensitivity_grapho',...
                                                    'specificity_alert','specificity_ripples','specificity_grapho',...
                                                    'f1_score_alert','f1_score_ripples','f1_score_grapho');

S.validation_filename = ['internal_test_64.mat'];


sensitivity_alert =[];specificity_alert =[];f1_score_alert =[];
sensitivity_ripples =[];specificity_ripples =[];f1_score_ripples =[];
sensitivity_grapho =[];specificity_grapho =[];f1_score_grapho =[];

for idx = 1:length(statstest_collec)
   
    sensitivity_alert(idx) = 100*statstest_collec(idx).sensitivity(1);
    sensitivity_ripples(idx) = 100*statstest_collec(idx).sensitivity(2);
    sensitivity_grapho(idx) = 100*statstest_collec(idx).sensitivity(3);
    
    specificity_alert(idx) = 100*statstest_collec(idx).specificity(1);
    specificity_ripples(idx) = 100*statstest_collec(idx).specificity(2);
    specificity_grapho(idx) = 100*statstest_collec(idx).specificity(3);
    
    f1_score_alert(idx) = statstest_collec(idx).Fscore(1);
    f1_score_ripples(idx) = statstest_collec(idx).Fscore(2);
    f1_score_grapho(idx) = statstest_collec(idx).Fscore(3);
  
end



save([S.validation_filepath S.validation_filename],'sensitivity_alert','sensitivity_ripples','sensitivity_grapho',...
                                                    'specificity_alert','specificity_ripples','specificity_grapho',...
                                                    'f1_score_alert','f1_score_ripples','f1_score_grapho');


end

temp =[];


