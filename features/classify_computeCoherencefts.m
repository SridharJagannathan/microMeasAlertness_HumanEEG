
function [coherencetab] = classify_computeCoherencefts(eegStruct,eleclabels)
% 
% Inputs: eegStruct   - EEGlab data structure,
%         eleclabels  - electrode labels,
%         
% Outputs: coherencetab - trial by trial coherence,
%
% classify_computeCoherencefts() - Performs the following
% Compute trial by trial coherence..
%_____________________________________________________________________________
% Author: Sridhar Jagannathan (27/09/2017).
%
% Copyright (C) 2017 Sridhar Jagannathan
%% Step 1: Initialize variables now..

coherencetab = [];
EEG = eegStruct;
%% Step 2: Convert to fieltrip..
data = eeglab2fieldtrip( EEG, 'preprocessing', 'none' );
data.label = data.label';

labelnames =fieldnames(eleclabels); chanlabels =[];

for idx = 1:length(labelnames)
    chanlabels = [chanlabels eleclabels.(labelnames{idx})];
    
end

%% Step 3: Compute the frequency spectrum..
cfg            = [];
cfg.output     = 'powandcsd';
cfg.method     = 'mtmfft';
cfg.foi          = 0.5:2:30;
cfg.tapsmofrq  = 5;
cfg.keeptrials = 'yes';
cfg.channel    = chanlabels;
cfg.channelcmb = {'all' 'all'}; 

evalexp = 'ft_freqanalysis(cfg, data);';
[T,freq] = evalc(evalexp);

%% Step 4: Compute the coherence..

cfg            = [];
cfg.channelcmb = {'all' 'all'}; 

tmpcmb = ft_channelcombination(cfg.channelcmb, data.label);
tmpchan = unique(tmpcmb(:));
cfg.channelcmb = ft_channelcombination(cfg.channelcmb, tmpchan, 1);

freq = ft_checkdata(freq, 'cmbrepresentation', 'sparse', 'channelcmb', cfg.channelcmb); 
powindx = labelcmb2indx(freq.labelcmb);

[nTrials,nChanComb,nFreq]=size(freq.crsspctrm);

coh = nan(nTrials,nChanComb,nFreq);
input = freq.crsspctrm;


for k = 1:nTrials
    
     p1    = squeeze(input(k,powindx(:,1),:));
     p2    = squeeze(input(k,powindx(:,2),:));
     denom = sqrt(p1.*p2);
    
     tmp    = (squeeze(abs(input(k,:,:)))./denom);
     tmp = tmp.^2;
     
     coh(k,:,:) = tmp;  
    
    
end

keepchn = powindx(:, 1) ~= powindx(:, 2);
coh = coh(:,keepchn, :);
freq.labelcmb = freq.labelcmb(keepchn, :);

channelcomb = freq.labelcmb;
freqlist = freq.freq;


%% Step 5: Compute individual band coherences..

deltafreq = [1 4]; %Frequency range in Hz 0.5-4
thetafreq = [4 7]; %Frequency range in Hz 4-7
alphafreq = [7 12]; %Frequency range in Hz 7-12
spindlefreq = [12 16]; %Frequency range in Hz 12-16
gammafreq = [16 30]; %Frequency range in Hz 16-30

[~, delfBeg] = min(abs(freqlist-deltafreq(1)));
[~, delfEnd] = min(abs(freqlist-deltafreq(2)));

[~, thefBeg] = min(abs(freqlist-thetafreq(1)));
[~, thefEnd] = min(abs(freqlist-thetafreq(2)));

[~, alpfBeg] = min(abs(freqlist-alphafreq(1)));
[~, alpfEnd] = min(abs(freqlist-alphafreq(2)));

[~, spinfBeg] = min(abs(freqlist-spindlefreq(1)));
[~, spinfEnd] = min(abs(freqlist-spindlefreq(2)));

[~, gammafBeg] = min(abs(freqlist-gammafreq(1)));
[~, gammafEnd] = min(abs(freqlist-gammafreq(2)));

delta_coh = mean(coh(:,:,delfBeg:delfEnd),3);
theta_coh = mean(coh(:,:,thefBeg:thefEnd),3);
alpha_coh = mean(coh(:,:,alpfBeg:alpfEnd),3);
spin_coh = mean(coh(:,:,spinfBeg:spinfEnd),3);
gamma_coh = mean(coh(:,:,gammafBeg:gammafEnd),3);

%% Step 6: Compute region pairs.. 

comb = eleclabels.frontal;
list_1 = match_str(channelcomb(:,1),comb);
list_2 = match_str(channelcomb(:,2),comb);
comm_list = intersect(list_1,list_2);

fronto_frontal.delta_coh = mean(delta_coh(:,comm_list),2);
fronto_frontal.theta_coh = mean(theta_coh(:,comm_list),2);
fronto_frontal.alpha_coh = mean(alpha_coh(:,comm_list),2);
fronto_frontal.spin_coh = mean(spin_coh(:,comm_list),2);
fronto_frontal.gamma_coh = mean(gamma_coh(:,comm_list),2);

%comb = {'C3', 'C4'};
comb = eleclabels.central;
list_1 = match_str(channelcomb(:,1),comb);
list_2 = match_str(channelcomb(:,2),comb);
comm_list = intersect(list_1,list_2);

centro_central.delta_coh = mean(delta_coh(:,comm_list),2);
centro_central.theta_coh = mean(theta_coh(:,comm_list),2);
centro_central.alpha_coh = mean(alpha_coh(:,comm_list),2);
centro_central.spin_coh = mean(spin_coh(:,comm_list),2);
centro_central.gamma_coh = mean(gamma_coh(:,comm_list),2);

%comb = {'T7', 'T8'};
comb = eleclabels.temporal;
list_1 = match_str(channelcomb(:,1),comb);
list_2 = match_str(channelcomb(:,2),comb);
comm_list = intersect(list_1,list_2);

temporo_temporal.delta_coh = mean(delta_coh(:,comm_list),2);
temporo_temporal.theta_coh = mean(theta_coh(:,comm_list),2);
temporo_temporal.alpha_coh = mean(alpha_coh(:,comm_list),2);
temporo_temporal.spin_coh = mean(spin_coh(:,comm_list),2);
temporo_temporal.gamma_coh = mean(gamma_coh(:,comm_list),2);

%comb = {'Oz','O1','O2'};
comb = eleclabels.occipetal;
list_1 = match_str(channelcomb(:,1),comb);
list_2 = match_str(channelcomb(:,2),comb);
comm_list = intersect(list_1,list_2);

occipeto_occipetal.delta_coh = mean(delta_coh(:,comm_list),2);
occipeto_occipetal.theta_coh = mean(theta_coh(:,comm_list),2);
occipeto_occipetal.alpha_coh = mean(alpha_coh(:,comm_list),2);
occipeto_occipetal.spin_coh = mean(spin_coh(:,comm_list),2);
occipeto_occipetal.gamma_coh = mean(gamma_coh(:,comm_list),2);

%% Step 7: summarise return values

dummyvals =  {'---','---','---','---','---',...
              '---','---','---','---','---',...
              '---','---','---','---','---',...
              '---','---','---','---','---'};
dummycell = repmat(dummyvals,nTrials,1);

coherencetab = cell2table(dummycell,...
    'VariableNames',{'Frontal_Alpha','Frontal_Theta','Frontal_Delta','Frontal_Spin','Frontal_Gamma',...
                     'Central_Alpha','Central_Theta','Central_Delta','Central_Spin','Central_Gamma',...
                     'Temporal_Alpha','Temporal_Theta','Temporal_Delta','Temporal_Spin','Temporal_Gamma',...
                     'Occipetal_Alpha','Occipetal_Theta','Occipetal_Delta','Occipetal_Spin','Occipetal_Gamma'});

coherencetab.Frontal_Alpha = fronto_frontal.alpha_coh;
coherencetab.Frontal_Theta = fronto_frontal.theta_coh;
coherencetab.Frontal_Delta = fronto_frontal.delta_coh;
coherencetab.Frontal_Spin = fronto_frontal.spin_coh;
coherencetab.Frontal_Gamma = fronto_frontal.gamma_coh;

coherencetab.Central_Alpha = centro_central.alpha_coh;
coherencetab.Central_Theta = centro_central.theta_coh;
coherencetab.Central_Delta = centro_central.delta_coh;
coherencetab.Central_Spin = centro_central.spin_coh;
coherencetab.Central_Gamma = centro_central.gamma_coh;

coherencetab.Temporal_Alpha = temporo_temporal.alpha_coh;
coherencetab.Temporal_Theta = temporo_temporal.theta_coh;
coherencetab.Temporal_Delta = temporo_temporal.delta_coh;
coherencetab.Temporal_Spin = temporo_temporal.spin_coh;
coherencetab.Temporal_Gamma = temporo_temporal.gamma_coh;

coherencetab.Occipetal_Alpha = occipeto_occipetal.alpha_coh;
coherencetab.Occipetal_Theta = occipeto_occipetal.theta_coh;
coherencetab.Occipetal_Delta = occipeto_occipetal.delta_coh;
coherencetab.Occipetal_Spin = occipeto_occipetal.spin_coh;
coherencetab.Occipetal_Gamma = occipeto_occipetal.gamma_coh;



end
