
function [Vertex, Vertex_ft] = classify_computeVertexBiphasicfts(EEGdata,Freq_sample)
% 
% classify_computeVertexBiphasicfts() - Performs the following
% Comoutes features for biphasic vertexes in each trial for a particular channel..
%_____________________________________________________________________________
% Author: Sridhar Jagannathan (27/09/2017).
%
% Copyright (C) 2017 Sridhar Jagannathan
%

Vertex =[]; Vertex_ft = [];
%% Vertex parameters..
Vertex.freqrange = 0.25:0.25:6; % frequencies over which to search for the vertexes 2:4;
Freq_begin = Vertex.freqrange(1);Freq_end = Vertex.freqrange(end);
Vertex.min_duration = 0.01; Vertex.max_duration = 1.5; % expected vertex duration in seconds

%% Resampling and other parameters..
Freq_resample = 100;  %Resample the data so you can get everything standardized..

%% Step1: Resample the original EEG signal..
EEGdata = resample(EEGdata,Freq_resample,Freq_sample);

%% Step2: Filter the original EEG signal..
evalexp = 'eegfiltfft(EEGdata, Freq_resample, Freq_begin, Freq_end);';
[T,EEGdata] = evalc(evalexp);

%% Step3: Set up tunable parameters..
Vertex.maximathresh = 40; %threshold for minima -40uV..
Vertex.pkdist=1.5*Freq_resample;%peak distance seperation 1.5*100 = 150 samples -> 1.5 secs
if length(EEGdata) <= Vertex.pkdist %For signals with shorter time duration..
   Vertex.pkdist=length(EEGdata)-2; 
end
Vertex.scantime = 0.50*Freq_resample; % scan time for a positive peak in secs..-> 0.5 secs

%% Step4: Apply the actual transient finder..
chan_data = double(EEGdata);
MinScaledData = chan_data -min(chan_data);
        
%Find maximas wrt to minima in the data now..
[pks,locs] = findpeaks(MinScaledData,'MINPEAKDISTANCE',Vertex.pkdist);
Maxima = chan_data(locs); %find out actual minimas..

%Check if the minimas drop below the threshold..
mm=Maxima(Maxima>=Vertex.maximathresh);
[~,pos]= ismember(mm,Maxima);

pospks =[]; poslocs =[];
negpks =[]; neglocs =[];
                
%Check if the value is actually stand out..
indk=locs(pos); 
nbr_kc=length(indk);
        
for i=1:length(indk) 
    timerange = min(indk(i) + Vertex.scantime, length(MinScaledData));
    negpeakdata = -chan_data(indk(i)+1:timerange);
    if length(negpeakdata)>=3
    [inegpks,ineglocs] = findpeaks(negpeakdata,'NPeaks',1,'SortStr','descend');
    else
        inegpks =[];
    end

    pospks(i)  = chan_data(indk(i));
    poslocs(i) = indk(i);
    if ~isempty(inegpks)
    negpks(i)  = chan_data(ineglocs+ indk(i));
    neglocs(i) = ineglocs+ indk(i);
    else
        negpks(i) = 0;
        neglocs(i) = 0;
    end

end
        
       
%% Step4: Set up the return values now..
        
        vertexfts=[];
        
        %vertexfts.duration =[pos_kc(1)];
        vertexfts.count =[nbr_kc];
        %vertexfts.negpk =knegpk;
        
        [~,Idxppk] = max(pospks);
        [~,Idxnpk] = min(negpks);
        
        vertexfts.negpks = min(negpks);
        vertexfts.pospks = max(pospks);
        vertexfts.poslocs = poslocs(Idxppk);
        vertexfts.neglocs = neglocs(Idxnpk);
        vertexfts.duration =[vertexfts.neglocs - vertexfts.poslocs]/Freq_resample;
   
  
%% Step5: Reduce the vertex by duration characteristics..
          if(vertexfts.duration>Vertex.max_duration)
              vertexfts.count = 0; 
              
          end
          if(vertexfts.duration<Vertex.min_duration)
              vertexfts.count = 0; 
              
          end
%% Step6: Organize the data for each vertex detection..
Vertex_ft = vertexfts;
Vertex = vertexfts;
                
end
