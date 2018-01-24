%% Start afresh by clearing all command windows and variables
%clc; 
clear;

rng(1); % For reproducibility

%% Load the paths..
loadpath;

%% Load the input files..
subject_ids = {'merged'};

rmsubject_ids = {'','5','7','8','9','13','15', ...
               '17','19','20','21','23','24', ...
               '26','27','28','29','30','36', ...
               '41','42','44','45','48','49', ...
               '51','53','54','55','56','58', ...
               '60'};
for k = 1:1 %length(rmsubject_ids)         
           
for m = 1:length(subject_ids)
    
subject = subject_ids{m};
   
%1. Preprocessed file -- > This contains the EEGlab preprocessed file
S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
S.eeg_filename = [subject '_pretrial_preprocess'];

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);



%2. Load the class labels..Horiscale data  --> Common for all subjects
S.hori_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess/'];
S.hori_filename = 'merged_AudMaskingCz_hori.mat';

hori_data = load([S.hori_filepath S.hori_filename]);
Goldscore = hori_data.horidataset.HoridataV;
allsubj_id = hori_data.horidataset.subj_id;

%3. Choose the usable labels..
na_trls = find(isnan(Goldscore));
rm_subjs = find(allsubj_id == str2double(rmsubject_ids(k)));

fprintf('\n--Removing from the model: subj_%d--\n',str2double(rmsubject_ids(k)));

 %rm_trls = union(nc_trls,na_trls);
 %rm_trls = union(rm_trls,rm_subjs);
 rm_trls =[];

Gold_usable = Goldscore;
%Gold_usable(rm_trls)=[];

indexes = nan(length(Gold_usable),1);
alert_trls = find(Gold_usable<=2);
ripple_trls = intersect(find(Gold_usable>=3),find(Gold_usable<=5));
grapho_trls = find(Gold_usable>=6);

indexes(alert_trls) = 1;
indexes(ripple_trls) = 2;
indexes(grapho_trls) = 2;



graphoindexes = nan(length(Gold_usable),1);
graphoindexes(ripple_trls) = 2;
graphoindexes(grapho_trls) = 3;


S.feat_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/Scripts/classify_libsvm/'];
S.feat_filename = ['feat_externalvaldasgroups.mat'];

if ~exist([S.feat_filepath S.feat_filename], 'file')   %Check if features have been computed

evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

%load the preprocessed EEGdata set..
[T,EEG] = evalc(evalexp);
    
    
evalexp = 'pop_select(EEG, ''notrial'', rm_trls)';

%load the usable EEGdata set..
[T,EEG] = evalc(evalexp);

%% Use only some channels ..
electrodes_rx = {'E75','E70','E83',...
                 'E35', 'E110', ...
                 'E90',...
                 'E40','E109','E101','E115','E100',...
                 'E27', 'E123', 'E11'};    
    
chanlabels={EEG.chanlocs.labels};
selec_elec = ismember(chanlabels,electrodes_rx);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_rx] = evalc(evalexp);

EEG = EEG_rx;


%Collect channel labels..
chanlabels={EEG.chanlocs.labels};

electrodes_occ = {'E75','E70','E83'};
electrodes_tempero = {'E109','E101','E115','E100'};
electrodes_frontal = {'E27', 'E123', 'E11'};
electrodes_central = {'E35', 'E110'};
electrodes_parietal = {'E90'};

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

[trials_alert, misc_alert]= classify_detectAlertTrials(EEG_occ);


eleclabels.frontal = {'E27', 'E123', 'E11'};
eleclabels.central = {'E35', 'E110'};
eleclabels.parietal = {'E62'};
eleclabels.temporal =  {'E40','E109'};
eleclabels.occipetal = {'E75','E70','E83'};

[coh]= classify_detectCoherenceTrials(EEG,eleclabels);
coh_features = table2array(coh);6064+7229+481
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
6064+7229+481
    
else
    load([S.feat_filepath S.feat_filename]);
end


end

%% Arrange the features and labels..
               
feat_groups{1,m} = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];

class_feats = [indexes feat_groups{1,1}];

 rm_trls = na_trls;
%  rm_trls = union(nc_trls,na_trls);
%  rm_trls = union(rm_trls,rm_subjs);
 %rm_trls = union(rm_trls,nb_trls);
 
 
 class_feats(rm_trls,:) = [];
 grapho_feats = [graphoindexes grapho_comb];
 grapho_feats(rm_trls,:) = [];
 
 spind_feats = [graphoindexes spind_comb];
 spind_feats(rm_trls,:) = [];
 
 nonspind_feats = [graphoindexes nonspind_comb];
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
statsexterntrain_collec=[];
statsexterntest_collec=[];

clear model_collecAlert minimums_collAlert ranges_collAlert
clear model_collecGrapho minimums_collGrapho ranges_collGrapho


S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/Scripts/classify_libsvm/'];
%S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/microMeasAlertness_HumanEEG_old/models/'];

S.model_filename = ['model_collec64_' char(rmsubject_ids(k)) ];

load([S.model_filepath S.model_filename]);

  bestModelAlert = model_collecAlert(1);
  bestModelGrapho = model_collecGrapho(1);
        
 %To scale the training and testing data in the same way..
 minimumsAlert = minimums_collAlert(1,:);
 rangesAlert = ranges_collAlert(1,:);
 
 minimumsGrapho = minimums_collGrapho(1,:);
 rangesGrapho = ranges_collGrapho(1,:);

fprintf('\n--using %s--\n',string(S.model_filename)); 

 for kparts = 1:c.NumTestSets


trainIdx = c.training(kparts);
testIdx = c.test(kparts);

trainData = DataSet(find(trainIdx),:);
trainLabel = DataClass(find(trainIdx));

testData = DataSet(find(testIdx),:);
testLabel = DataClass(find(testIdx));


GraphotrainData = GraphoDataSet(find(trainIdx),:);
GraphotrainLabel = GraphoDataClass(find(trainIdx));

GraphotestData = GraphoDataSet(find(testIdx),:);
GraphotestLabel = GraphoDataClass(find(testIdx));

SpindtrainData = SpindDataSet(find(trainIdx),:);
SpindtrainLabel = SpindDataClass(find(trainIdx));

SpindtestData = SpindDataSet(find(testIdx),:);
SpindtestLabel = SpindDataClass(find(testIdx));

nonSpindtrainData = nonSpindDataSet(find(trainIdx),:);
nonSpindtrainLabel = nonSpindDataClass(find(trainIdx));

nonSpindtestData = nonSpindDataSet(find(testIdx),:);
nonSpindtestLabel = nonSpindDataClass(find(testIdx));


% trainData = DataSet; 
% trainLabel = DataClass; 


%To scale the training and testing data in the same way..
OrigtrainData_class = trainData;
trainData = (trainData - repmat(minimumsAlert, size(trainData, 1), 1)) ./ repmat(rangesAlert, size(trainData, 1), 1);
OrigtestData_class = testData;
testData = (testData - repmat(minimumsAlert, size(testData, 1), 1)) ./ repmat(rangesAlert, size(testData, 1), 1);

[predictedLabel, accuracy, prob_values] = svmpredict(trainLabel, trainData, bestModelAlert, '-b 1');

[confusionMatrixtrain,ordertrain] = confusionmat(trainLabel,predictedLabel);

spind_poss = intersect(find(predictedLabel == 2),find(SpindtrainData==1));
nonspind_poss = intersect(find(predictedLabel == 2),find(nonSpindtrainData==1));

subtraindatagrapho = OrigtrainData_class(spind_poss,:);
subtrainLabelgrapho = trainLabel(spind_poss);
       
subtraindatagrapho = (subtraindatagrapho - repmat(minimumsGrapho, size(subtraindatagrapho, 1), 1)) ./ repmat(rangesGrapho, size(subtraindatagrapho, 1), 1);

[subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtrainLabelgrapho, subtraindatagrapho, bestModelGrapho, '-b 1');
       
predactualgrapho = find(subpredictedLabelgrapho ==3);
predictedLabel(spind_poss(predactualgrapho)) = 3;
predictedLabel(nonspind_poss) = 3;


 grapho_actual= intersect(find(trainLabel==2),find(GraphotrainLabel==3));
 trainLabel(grapho_actual) = 3;
 
 % grapho_pred = intersect(find(predictedLabel==2),find(GraphotrainData==1));
% predictedLabel(grapho_pred) = 3;

[confusionMatrixAll,orderAll] = confusionmat(trainLabel,predictedLabel);


stats_train = confusionmatStats(trainLabel,predictedLabel);



% figure; imagesc(confusionMatrixAll');
% xlabel('actual class label');
% ylabel('predicted class label');
% title(['confusion matrix for overall classification']);
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
  [predictedLabel, accuracy, decisValueWinner] = svmpredict(testLabel, testData, bestModelAlert, '-b 1'); % run the SVM model on the test data
  
  spind_poss = intersect(find(predictedLabel == 2),find(SpindtestData==1));
  nonspind_poss = intersect(find(predictedLabel == 2),find(nonSpindtestData==1));

  subtestdatagrapho = OrigtestData_class(spind_poss,:);
  subtestLabelgrapho = testLabel(spind_poss);
  
  subtestdatagrapho = (subtestdatagrapho - repmat(minimumsGrapho, size(subtestdatagrapho, 1), 1)) ./ repmat(rangesGrapho, size(subtestdatagrapho, 1), 1);

 [subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, '-b 1');
       
  predactualgrapho = find(subpredictedLabelgrapho ==3);
  predictedLabel(spind_poss(predactualgrapho)) = 3;
  predictedLabel(nonspind_poss) = 3;
  
  
  grapho_actual= intersect(find(testLabel==2),find(GraphotestLabel==3));
  testLabel(grapho_actual) = 3;
  
%   grapho_pred = intersect(find(predictedLabel==2),find(GraphotestData==1));
%   predictedLabel(grapho_pred) = 3;
  
  
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


statsexterntrain_collec = [statsexterntrain_collec stats_train];
statsexterntest_collec = [statsexterntest_collec stats_test];
%acctrain_collec=[acctrain_collec accuracyAll*100];
%acctest_collec=[acctest_collec accuracytest*100];

end

% S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/Scripts/classify_libsvm/'];
% S.model_filename = ['model_collec128_' char(rmsubject_ids(k)) ];
% 
% save([S.model_filepath S.model_filename],'model_collecAlert','ranges_collAlert','minimums_collAlert',...
%                                          'model_collecGrapho','ranges_collGrapho','minimums_collGrapho',...
%                                          'statstrain_collec','statstest_collec');
                                     
                                     
S.validation_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/validation/'];
S.validation_filename = ['externalvaldas_train_64.mat'];
sensitivity_alert =[];specificity_alert =[];f1_score_alert =[];
sensitivity_ripples =[];specificity_ripples =[];f1_score_ripples =[];
sensitivity_grapho =[];specificity_grapho =[];f1_score_grapho =[];

for idx = 1:length(statsexterntrain_collec)
   
    sensitivity_alert(idx) = 100*statsexterntrain_collec(idx).sensitivity(1);
    sensitivity_ripples(idx) = 100*statsexterntrain_collec(idx).sensitivity(2);
    sensitivity_grapho(idx) = 100*statsexterntrain_collec(idx).sensitivity(3);
    
    specificity_alert(idx) = 100*statsexterntrain_collec(idx).specificity(1);
    specificity_ripples(idx) =100*statsexterntrain_collec(idx).specificity(2);
    specificity_grapho(idx) = 100*statsexterntrain_collec(idx).specificity(3);
    
    f1_score_alert(idx) = statsexterntrain_collec(idx).Fscore(1);
    f1_score_ripples(idx) = statsexterntrain_collec(idx).Fscore(2);
    f1_score_grapho(idx) = statsexterntrain_collec(idx).Fscore(3);
  
end



save([S.validation_filepath S.validation_filename],'sensitivity_alert','sensitivity_ripples','sensitivity_grapho',...
                                                    'specificity_alert','specificity_ripples','specificity_grapho',...
                                                    'f1_score_alert','f1_score_ripples','f1_score_grapho');

S.validation_filename = ['externalvaldas_test_64.mat'];


sensitivity_alert =[];specificity_alert =[];f1_score_alert =[];
sensitivity_ripples =[];specificity_ripples =[];f1_score_ripples =[];
sensitivity_grapho =[];specificity_grapho =[];f1_score_grapho =[];

for idx = 1:length(statsexterntest_collec)
   
    sensitivity_alert(idx) = 100*statsexterntest_collec(idx).sensitivity(1);
    sensitivity_ripples(idx) = 100*statsexterntest_collec(idx).sensitivity(2);
    sensitivity_grapho(idx) = 100*statsexterntest_collec(idx).sensitivity(3);
    
    specificity_alert(idx) = 100*statsexterntest_collec(idx).specificity(1);
    specificity_ripples(idx) = 100*statsexterntest_collec(idx).specificity(2);
    specificity_grapho(idx) = 100*statsexterntest_collec(idx).specificity(3);
    
    f1_score_alert(idx) = statsexterntest_collec(idx).Fscore(1);
    f1_score_ripples(idx) = statsexterntest_collec(idx).Fscore(2);
    f1_score_grapho(idx) = statsexterntest_collec(idx).Fscore(3);
  
end



save([S.validation_filepath S.validation_filename],'sensitivity_alert','sensitivity_ripples','sensitivity_grapho',...
                                                    'specificity_alert','specificity_ripples','specificity_grapho',...
                                                    'f1_score_alert','f1_score_ripples','f1_score_grapho');
 

end

temp =[];


