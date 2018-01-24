
function [trialstruc] = classify_microMeasures(eegStruct, modelfilepath)
% 
% Inputs: EEGlabStruct   - EEGlab data structure,
%         modelfilepath   - Path of the trained model,
% Outputs: trialstruc   - Trial indexs of alert, drowsymild,
%         drowsysevere, and also vertex, spindle, k-complex indices..
%
% classify_microMeasures() - Performs the following
% Produces a classification of trials as alert, drowsymild or drowsysevere..
%
% Requirements: EEGlab (tested with eeglab13_5_4b), fieldtrip (tested with
% fieldtrip-20151223), Matlab (tested with R2016b)
%_____________________________________________________________________________
% Author: Sridhar Jagannathan (23/10/2017).
%
% Copyright (C) 2017 Sridhar Jagannathan
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
%%
trialstruc = [];
EEG = eegStruct;

%% Step1: Use only some channels ..
           
if eegStruct.nbchan>=70
    electrodes_rx = {'E75','E70','E83',...
                     'E36', 'E104', ... 
                     'E90',...
                     'E45','E108','E102','E114','E100',...
                     'E33', 'E122', 'E11'};    
else
    
    electrodes_rx = {'Oz','O1','O2',...
                     'C3', 'C4', ...
                     'PO10',...
                     'T7','T8','TP8','FT10','TP10',...
                     'F7', 'F8', 'Fz'};
    
end
            
chanlabels={EEG.chanlocs.labels};
selec_elec = ismember(chanlabels,electrodes_rx);
remove_elec = find(~selec_elec);%Use only selected electrodes..
evalexp = 'pop_select(EEG,''nochannel'',remove_elec);';
[T,EEG_rx] = evalc(evalexp);

EEG = EEG_rx;

%% Step2: Produce reduced channels now....

%Collect channel labels..
chanlabels={EEG.chanlocs.labels};

if eegStruct.nbchan>=70
    
    electrodes_occ = {'E75','E70','E83'};
    electrodes_tempero = {'E108','E102','E114','E100'};
    electrodes_frontal = {'E33', 'E122', 'E11'};
    electrodes_central = {'E36', 'E104'};
    electrodes_parietal = {'E90'};
    
else
    
    electrodes_occ = {'Oz','O1','O2'};
    electrodes_tempero = {'T8','TP8','FT10','TP10'};
    electrodes_frontal = {'F7', 'F8', 'Fz'};
    electrodes_central = {'C3', 'C4'};
    electrodes_parietal = {'PO10'};
    
end



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

%% Step3: Computing Alert features..

fprintf('\n--Computing Variance features--\n');    

[trials_alert, misc_alert]= classify_computeVariancefts(EEG_occ);

if eegStruct.nbchan>=70
    
    eleclabels.frontal = {'E33', 'E122', 'E11'};
    eleclabels.central = {'E36', 'E104'};
    eleclabels.parietal = {'E62'};
    eleclabels.temporal =  {'E45','E108'};
    eleclabels.occipetal = {'E75','E70','E83'};
    
else
    
    eleclabels.frontal = {'F7', 'F8', 'Fz'};
    eleclabels.central = {'C3', 'C4'};
    eleclabels.parietal = {'Pz'};
    eleclabels.temporal =  {'T7', 'T8'};
    eleclabels.occipetal = {'Oz','O1', 'O2'};
    
end



fprintf('\n--Computing Coherence features--\n');    

[coh]= classify_computeCoherencefts(EEG,eleclabels);
coh_features = table2array(coh);

%% Step4: Computing Vertex features..
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

%% Step5: Computing spindle features..
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

%% Step6: Computing k-complex features..

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

%% Step7: Combine graphical elements..
vertex_def = union(biphasic_def,monophasic_def);
grapho_def = union(vertex_def,kcomp_def);
nonspind_def = grapho_def;

spind_comb = zeros(EEG.trials,1);
spind_comb(spindle_def) = 1;

nonspind_comb = zeros(EEG.trials,1);
nonspind_comb(nonspind_def) = 1;

%% Step8: Summarize the features..
testData = [misc_alert.varian.freqband2' misc_alert.varian.freqband5' misc_alert.varian.freqband6'...
                    misc_alert.varian.freqband10'...
                    coh_features];

%% Step9: Load the model..

clear model_collecAlert minimums_collAlert ranges_collAlert
clear model_collecGrapho minimums_collGrapho ranges_collGrapho

load(modelfilepath);

idx = 1;
bestModelAlert = model_collecAlert(idx);
bestModelGrapho = model_collecGrapho(idx);

%To scale the training and testing data in the same way..
minimumsAlert = minimums_collAlert(idx,:);
rangesAlert = ranges_collAlert(idx,:);

minimumsGrapho = minimums_collGrapho(idx,:);
rangesGrapho = ranges_collGrapho(idx,:);

%% Step10: Predict using the model..
%scale the testing data now..
OrigtestData_class = testData;
testData_class = (testData - repmat(minimumsAlert, size(testData, 1), 1)) ./ repmat(rangesAlert, size(testData, 1), 1);

[text, predict_label, accuracy, prob_values] = evalc('svmpredict(zeros(length(testData_class),1), testData_class, bestModelAlert, ''-b 1'');');
        
spind_poss = intersect(find(predict_label == 2),find(spind_comb==1));
nonspind_poss = intersect(find(predict_label == 2),find(nonspind_comb==1));
       
subtestdatagrapho = OrigtestData_class(spind_poss,:);
subtestLabelgrapho = zeros(length(subtestdatagrapho),1);
       
subtestdatagrapho = (subtestdatagrapho - repmat(minimumsGrapho, size(subtestdatagrapho, 1), 1)) ./...
                    repmat(rangesGrapho, size(subtestdatagrapho, 1), 1);


[text, subpredictedLabelgrapho, accuracy, decisValueWinner] = evalc('svmpredict(subtestLabelgrapho, subtestdatagrapho, bestModelGrapho, ''-b 1'');');

predactualgrapho = find(subpredictedLabelgrapho ==3);
predict_label(spind_poss(predactualgrapho)) = 3;
predict_label(nonspind_poss) = 3;
              
%% Step11: Summarise the trials now..
trialstruc.alert = find(predict_label == 1);
trialstruc.drowsymild = find(predict_label == 2);
trialstruc.drowsysevere = find(predict_label == 3);
trialstruc.vertgrapho = intersect(trialstruc.drowsysevere,vertex_def);
trialstruc.spingrapho = intersect(trialstruc.drowsysevere,spindle_def);
trialstruc.kcompgrapho = intersect(trialstruc.drowsysevere,kcomp_def);

end