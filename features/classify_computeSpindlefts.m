
function [Spindle, Spindle_ft] = classify_computeSpindlefts(EEGdata,Freq_sample,Freq_range,Time_params)
% 
% classify_computeSpindlefts() - Performs the following
% Computes features for spindles in each trial for a particular channel..
%_____________________________________________________________________________
% Author: Sridhar Jagannathan (27/09/2017).
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
Spindle =[]; Spindle_ft = [];
%% Spindle parameters..
if nargin<3 || isempty(Freq_range), 
    Spindle.freqrange = 11:16; % spindle frequencies over which to search for the spindles 13.5:16;11:16;
else
    Spindle.freqrange = Freq_range;
end
if nargin<4 || isempty(Time_params), 
    Spindle.min_duration = 0.4; Spindle.max_duration = 1.5; % expected spindle duration in seconds(0.4,1.5)
else
    Spindle.min_duration = Time_params(1); Spindle.max_duration = Time_params(2);
end
Freq_begin = Spindle.freqrange(1);Freq_end = Spindle.freqrange(end);


%% Resampling and other parameters..
Freq_resample = 100;  %Resample the data so you can get everything standardized..
windowlen = 0.12*Freq_resample; %Number of time samples to be used in the windowed mean calculation..

%% Step1: Resample the original EEG signal..
EEGdata = resample(EEGdata,Freq_resample,Freq_sample);
rescale = Freq_resample/Freq_sample; % rescale the time parameter..


%% Step2: Scale and wavelet paramters..
wtype = 'morl'; % Morlet wavelet
scales = 2:0.1:15; % scales corresponding to pseudo-frequencies
EEGdata = double(EEGdata);


%% Step3: Apply the actual CWT..
        
        chan_data = EEGdata;
        
        wavelet = cwt(chan_data,scales,wtype); % Apply CWT
        
        wavelet_mod = abs(wavelet.*wavelet); 
        normwavelet = 100*wavelet_mod./sum(wavelet_mod(:)); 
        
        Freq_pseudo = scal2frq(scales,wtype,1/Freq_resample); 
        [~,Scale_end] = min(abs(Freq_pseudo-Freq_begin));
        [~,Scale_begin] = min(abs(Freq_pseudo-Freq_end));
        scales_member = Scale_begin:Scale_end;
        

        [~, sortedIdx] = sort(normwavelet,'ascend'); %Sort according to the power..
        maxrank = size(sortedIdx,1):-1:size(sortedIdx,1)-length(scales_member)-1; %Compute the max possible ranks available..
        
        spindlfts=[];
        
            for k = 1:size(sortedIdx,2) %For each timepoint compute the rank for each member..
                [~, isortedIdx{k,1}, ~] = intersect(sortedIdx(:,k), scales_member);
                spindlfts.rankwavlet(1,k) = sum(isortedIdx{k});
                spindlfts.rankwavlet(2,k) = sum(isortedIdx{k})/sum(maxrank);
                spindlfts.rankprob(k) = spindlfts.rankwavlet(2,k);
                
                % now try to use the normalized ranks as chances for spindles to occur
                spindlfts.prob(k) = spindlfts.rankprob(k);
            end
            
        
        spindlfts.prob_slidwin = movmean(spindlfts.prob, windowlen,'includenan','Endpoints','fill');
        
        %spindlfts.prob_slidwin = spindlfts.prob_slidwin(:)';
        spindlfts.probmax = max(spindlfts.prob, spindlfts.prob_slidwin); %Just to increase sensitivity..
        
        [Spindlelocs] = find(spindlfts.probmax>0.6)+0;
        
        start = []; stop = [];
        if(Spindlelocs~=0)
            % Now, find successive differences of spindle prob candidates
            d = [0, abs(diff(Spindlelocs))]; crosspts = find(d>1);
            start = [Spindlelocs(1), Spindlelocs(crosspts)]; % start locs..
            stop = [Spindlelocs(crosspts-1), Spindlelocs(end)]; % stop locs..
        end
  
%% Step4: Reduce the spindles by duration characteristics..
        Spindledur = abs(start - stop); % duration
        rmspindles = find(Spindledur<Spindle.min_duration*Freq_resample);
        rmspindles = [rmspindles find(Spindledur>Spindle.max_duration*Freq_resample)];
        start(rmspindles) = []; stop(rmspindles) = [];
%% Step5: Organize the data for each spindle detection..
        for i = 1:length(start)
            Spindle_ft{i}.index = i;
            Spindle_ft{i}.duration = (round(stop(i) - start(i) +1))/Freq_resample;
            Spindle_ft{i}.prob = mean(spindlfts.probmax(start(i):stop(i)));
            Spindle_ft{i}.meanbg = mean(spindlfts.prob);
            tempdata = EEGdata(start(i):stop(i));
            [Spindle_ft{i}.pospks,Spindle_ft{i}.poslocs]=findpeaks(tempdata,'MinPeakProminence',5); 
            [Spindle_ft{i}.negpks,Spindle_ft{i}.neglocs]=findpeaks(-tempdata,'MinPeakProminence',5); 
            Spindle_ft{i}.meanpospks = mean(Spindle_ft{i}.pospks);
            Spindle_ft{i}.lenpospks = length(Spindle_ft{i}.pospks);
            Spindle_ft{i}.meannegpks = mean(Spindle_ft{i}.negpks);
            Spindle_ft{i}.lennegpks = length(Spindle_ft{i}.negpks);
            Spindle_ft{i}.maxval = max(tempdata);
            Spindle_ft{i}.minval = min(tempdata);
            Spindle_ft{i}.crossing = crossing(tempdata,[],mean(tempdata));
        end
        

%return items..
Spindle.start_stop = [start(:), stop(:)]./rescale;
        
        
        
end