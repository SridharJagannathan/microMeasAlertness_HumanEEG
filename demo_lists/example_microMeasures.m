clear
%clc
%close all

pathappend = '/work/imagingQ/';

%% Add paths now..
%SPM path
spm_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/spm12'];
addpath(spm_toolbox);


%Fieldtrip path
ftp_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/fieldtrip-20151223'];
addpath(ftp_toolbox);
%ft_defaults;

%EEGlab path
eeglab_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/eeglab13_5_4b'];
addpath(genpath(eeglab_toolbox));

%lib-SVM path
svm_toolbox = [pathappend 'SpatialAttention_Drowsiness/Scripts/toolboxes/libsvm-3.12'];
addpath(genpath(svm_toolbox));

% addpath to the data
dirData = [svm_toolbox]; 
addpath(dirData);

subject_ids = {'5'};

           
for m =1:length(subject_ids)
           
subject = subject_ids{m};


% %1. Preprocessed file -- > This contains the EEGlab preprocessed file

S.eeg_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/valdas_maskingdataset/preprocess'];
S.eeg_filename = ['AuMa_' subject '_pretrial_preprocess'];

subjname = ['subj_' subject];

fprintf('\n--Processing :%s--\n',subjname);

%% load the preprocessed EEGdata set..
evalexp = 'pop_loadset(''filename'', [S.eeg_filename ''.set''], ''filepath'', S.eeg_filepath);';

[T,EEG] = evalc(evalexp);

%% Use only some channels ..
           
electrodes_rx = {'E75','E70','E83',...
                 'E36', 'E104', ... 
                 'E90',...
                 'E45','E108','E102','E114','E100',...
                 'E33', 'E122', 'E11'};        
            
chanlabels={EEG.chanlocs.labels};
selec_elec = ismember(chanlabels,electrodes_rx);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_rx] = evalc(evalexp);

EEG = EEG_rx;

%% Compute the features now..

%Collect channel labels..
chanlabels={EEG.chanlocs.labels};
electrodes_occ = {'E75','E70','E83'};
electrodes_tempero = {'E108','E102','E114','E100'};
electrodes_frontal = {'E33', 'E122', 'E11'};
electrodes_central = {'E36', 'E104'};
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

fprintf('\n--Computing Variance features--\n');    

[trials_alert, misc_alert]= classify_computeVariancefts(EEG_occ);


eleclabels.frontal = {'E33', 'E122', 'E11'};
eleclabels.central = {'E36', 'E104'};
eleclabels.parietal = {'E62'};
eleclabels.temporal =  {'E45','E108'};
eleclabels.occipetal = {'E75','E70','E83'};

fprintf('\n--Computing Coherence features--\n');    

[coh]= classify_computeCoherencefts(EEG,eleclabels);
coh_features = table2array(coh);
    
    
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
Freq_range = 12:16;Time_params = [0.4 1.5]; %0.8

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
grapho_def = union(biphasic_def,monophasic_def);
grapho_def = union(grapho_def,kcomp_def);
nonspind_def = grapho_def;

spind_comb = zeros(EEG.trials,1);
spind_comb(spindle_def) = 1;

nonspind_comb = zeros(EEG.trials,1);
nonspind_comb(nonspind_def) = 1;

%% summarize the features..
testData = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];
                
S.model_filepath = [ pathappend 'SpatialAttention_Drowsiness/SleepOnset_Classification/microMeasAlertness_HumanEEG/'];
                 
S.model_filename = ['model_collec64_'];
clear model_collecAlert minimums_collAlert ranges_collAlert
clear model_collecGrapho minimums_collGrapho ranges_collGrapho

load([S.model_filepath S.model_filename]);

idx = 1;
bestModelAlert = model_collecAlert(idx);
bestModelGrapho = model_collecGrapho(idx);

%To scale the training and testing data in the same way..
minimumsAlert = minimums_collAlert(idx,:);
rangesAlert = ranges_collAlert(idx,:);

minimumsGrapho = minimums_collGrapho(idx,:);
rangesGrapho = ranges_collGrapho(idx,:);

fprintf('\n--using %s--\n',string(S.model_filename));  
testsubj = subject;
fprintf('\n--Testing on subject:%s--\n',string(testsubj));

%scale the testing data now..
OrigtestData_class = testData;
testData_class = (testData - repmat(minimumsAlert, size(testData, 1), 1)) ./ repmat(rangesAlert, size(testData, 1), 1);
        
[predict_label, accuracy, prob_values] = svmpredict(zeros(length(testData_class),1), testData_class, bestModelAlert, '-b 1');
       
spind_poss = intersect(find(predict_label == 2),find(spind_comb==1));
nonspind_poss = intersect(find(predict_label == 2),find(nonspind_comb==1));
       
subtestdatagrapho = OrigtestData_class(spind_poss,:);
subtestLabelgrapho = zeros(length(subtestdatagrapho),1);
       
subtestdatagrapho = (subtestdatagrapho - repmat(minimumsGrapho, size(subtestdatagrapho, 1), 1)) ./...
                    repmat(rangesGrapho, size(subtestdatagrapho, 1), 1);


[subpredictedLabelgrapho, accuracy, decisValueWinner] = svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, '-b 1');

predactualgrapho = find(subpredictedLabelgrapho ==3);
predict_label(spind_poss(predactualgrapho)) = 3;
predict_label(nonspind_poss) = 3;

%% usage..
Alert_trls = find(predict_label == 1);
Drowsymild_trls = find(predict_label == 2);
Drowsysevere_trls = find(predict_label == 3);

      
   
 end


temp =[];