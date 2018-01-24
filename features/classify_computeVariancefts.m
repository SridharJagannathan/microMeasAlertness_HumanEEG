
function [trials,misc] = classify_computeVariancefts(eegStruct)
% 
% Inputs: EEGlabStruct   - EEGlab data structure,
%         
% Outputs: trials   - computes alert trial based on frequency band wins,
%          misc     - contains the actual variances that will be used later,
%
% classify_computeVariancefts() - Performs the following
% Parse only the trials that are alert..
% Divide data into frequency bins..
% Compute the variance explained for the different frequency bins..
% Use the combination 45, as it has the highest sensitivity for the 
% alert trials..
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
%% Step 1: Initialize variables now..

trials = []; 
misc = [];
EEG = eegStruct;
nelec = EEG.nbchan;
ntrials = EEG.trials;


S=[];

%Allocating frequency bands..
startfreq = 0;
endfreq = 20; %20
countvar = 1;
for k = startfreq:2:endfreq-2
    
    fieldname = ['freqband' num2str(countvar)];
    S.(fieldname)(1) = k;
    S.(fieldname)(2) = k+2;
    countvar = countvar + 1;
   
end

S.freqband1(1) = 0.5; %Just changing the first freq band alone..

% %Temp code..
S.freqband10(1) = 2;
S.freqband10(2) = 6;



bincount = length(fieldnames(S));


%Initializing dynamic variables now..
varian = []; maxim = []; tfreq = []; trls = []; trllist = []; bandnames =[]; countval = 1;
for k = 1:bincount
    
    fieldname = ['freqband' num2str(k)];
    varian.(fieldname) = deal(nan(nelec,ntrials));
    
    for m = 1:bincount
        if m ~= k
           tagname = ['freqband' num2str(k) '_' num2str(m)];
           bandnames{countval} = tagname; countval = countval + 1;
           maxim.(tagname) = deal(zeros(nelec,ntrials));
           trls.(tagname) = deal(zeros(1,ntrials));
        end
    end
    
    
end


%% Step 2: Compute band power and variance explained..

for k = 1:nelec
    
                
 % The function to have a spectrum for one electrode
 [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = ...
             newtimef(EEG.data(k,:,:), EEG.pnts,[EEG.xmin EEG.xmax]*1000, EEG.srate, 0, ...
              'padratio', 2, 'freqs', [0.5 40], ...
              'plotersp', 'off','plotitc','off','verbose','off');  
          
  Pow  = tfdata.*conj(tfdata); % power
  
  Pow = Pow(:,:,:);
          
  Fband = [];
  
  for n = 1:bincount
      tagnamebegin = ['Band' num2str(n) '_' 'fBeg'];
      tagnameend = ['Band' num2str(n) '_' 'fEnd'];
      fieldname = ['freqband' num2str(n)];
      
      [~, Fband.(tagnamebegin)] = min(abs(freqs-S.(fieldname)(1)));
      [~, Fband.(tagnameend)] =   min(abs(freqs-S.(fieldname)(2)));
     
           
  end
  

                              
 % compute power in a frequency band..
   power = [];
   
   for n = 1:bincount
       fieldname = ['freqband' num2str(n)];
       tagnamebegin = ['Band' num2str(n) '_' 'fBeg'];
       tagnameend = ['Band' num2str(n) '_' 'fEnd'];
       power.(fieldname) = squeeze(sum(Pow(Fband.(tagnamebegin):Fband.(tagnameend),:,:),1));
       
   end
   
   
   power_allFB = squeeze(sum(Pow(1:size(Pow,1),:,:),1));
   

 %variance explained by different bands..
     
   for n = 1:bincount
       
    fieldname = ['freqband' num2str(n)];
    varian.(fieldname)(k,:) = 100 - 100*var(power_allFB - power.(fieldname))./var(power_allFB);
       
   end
   
         
end



%% Step 3: Threshold the variances now..

for n = 1:bincount
    
    fieldname = ['freqband' num2str(n)];
    varian.(fieldname)(varian.(fieldname)<20) = 0;
    
    
end


%% Step 4: Find the favourable spatial points..

for n = 1:bincount
    
    fieldname_n = ['freqband' num2str(n)];
    
    for m = 1:bincount
         
        if m ~= n
           tagname = ['freqband' num2str(n) '_' num2str(m)];
           fieldname_m = ['freqband' num2str(m)];
           maxim.(tagname)(varian.(fieldname_n)>varian.(fieldname_m))=1;
           %maxim.(tagname)= maxim.(tagname).*varian.(fieldname_n);
           tfreq.(tagname) = sum(maxim.(tagname));
           
        end
    end


end


trllistlen =[];

for n = 1:bincount
    
    for m = 1:bincount
        
        if m ~= n
            tagname_front = ['freqband' num2str(n) '_' num2str(m)];
            tagname_last = ['freqband' num2str(m) '_' num2str(n)];
            trls.(tagname_front)(tfreq.(tagname_front)>tfreq.(tagname_last))=1;
            trllist.(tagname_front) = find((trls.(tagname_front))>0)';
            trllistlen = [trllistlen  length(trllist.(tagname_front))];
            
        end
        
    end
    
    
end


%% Step 5: Summarise the results now.. 

misc.triallistlen= trllistlen;
misc.bandnames = bandnames;
misc.S = S;

misc.varian = varian;


%trials.alert = trllist.freqband5_10;
trials.alert = trllist.freqband5_3;



end