
function [Kcomp, Kcomp_ft] = classify_computeKcomplexfts(EEGdata,Freq_sample)
% 
% classify_computeKcomplexfts() - Performs the following
% Comoutes features for kcomplexes in each trial for a particular channel..
%_____________________________________________________________________________
% Author: Sridhar Jagannathan (27/09/2017).
%
% Copyright (C) 2017 Sridhar Jagannathan
%%
Kcomp =[]; Kcomp_ft = [];
%% Kcomplex parameters..
Kcomp.freqrange = 0.25:0.25:6; %frequencies over which to search for the kcomplexes 2:4;
Freq_begin = Kcomp.freqrange(1);Freq_end = Kcomp.freqrange(end);
Kcomp.min_duration = 0.2; Kcomp.max_duration = 1.5; % expected kcomp duration in seconds

%% Resampling and other parameters..
Freq_resample = 100;  %Resample the data so you can get everything standardized..

%% Step1: Resample the original EEG signal..
EEGdata = resample(EEGdata,Freq_resample,Freq_sample);

%% Step2: Filter the original EEG signal..
evalexp = 'eegfiltfft(EEGdata, Freq_resample, Freq_begin, Freq_end);';
[T,EEGdata] = evalc(evalexp);

%% Step3: Set up tunable parameters..
Kcomp.minimathresh = -40; %threshold for minima -40uV..
Kcomp.pkdist=1.5*Freq_resample;%peak distance seperation 1.5*100 = 150 samples -> 1.5 secs
if length(EEGdata) <= Kcomp.pkdist %For signals with shorter time duration..
   Kcomp.pkdist=length(EEGdata)-2; 
end
Kcomp.scantime = 1*Freq_resample; % scan time for a positive peak in secs..-> 1 secs

%% Step4: Apply the actual transient finder..
chan_data = double(EEGdata);
MaxScaledData = max(chan_data) - chan_data; %scale the signal by the maximum value..
        
%Find minimas wrt to maxima in the data now..
[pks,locs] = findpeaks(MaxScaledData,'MINPEAKDISTANCE',Kcomp.pkdist);
Minima = chan_data(locs); %find out actual minimas..

%Check if the minimas drop below the threshold..
mm=Minima(Minima<=Kcomp.minimathresh);
[~,pos]= ismember(mm,Minima);

negpks =[]; neglocs =[];
pospks =[]; poslocs =[];
                
%Check if the value is actually stand out..
indk=locs(pos); 
nbr_kc=length(indk);
        
for i=1:length(indk) 
    timerange = min(indk(i) + Kcomp.scantime, length(MaxScaledData));
    pospeakdata = chan_data(indk(i)+1:timerange);
    if length(pospeakdata)>=3
    [ipospks,iposlocs] = findpeaks(pospeakdata,'NPeaks',1,'SortStr','descend');
    else
        ipospks =[];
    end

    negpks(i)  = chan_data(indk(i));
    neglocs(i) = indk(i);
    if ~isempty(ipospks)
    pospks(i)  = chan_data(iposlocs+ indk(i));
    poslocs(i) = iposlocs+ indk(i);
    else
        pospks(i) = 0;
        poslocs(i) = 0;
    end

end
        
       
%% Step4: Set up the return values now..
        
        kcompfts=[];
        
        %kcompfts.duration =[pos_kc(1)];
        kcompfts.count =[nbr_kc];
        %kcompfts.negpk =knegpk;
        
        [~,Idxnpk] = min(negpks);
        [~,Idxppk] = min(pospks);
        
        kcompfts.negpks = min(negpks);
        kcompfts.pospks = max(pospks);
        kcompfts.poslocs = poslocs(Idxnpk);
        kcompfts.neglocs = neglocs(Idxnpk);
        kcompfts.duration =[kcompfts.poslocs-kcompfts.neglocs]/Freq_resample;
   
  
%% Step5: Reduce the kcomplex by duration characteristics..
          if(kcompfts.duration>Kcomp.max_duration)
              kcompfts.count = 0; 
              
          end
          if(kcompfts.duration<Kcomp.min_duration)
              kcompfts.count = 0; 
              
          end
%% Step6: Organize the data for each kcomplex detection..
Kcomp_ft = kcompfts;
Kcomp = kcompfts;
                
end
