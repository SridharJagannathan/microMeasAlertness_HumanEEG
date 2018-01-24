The project is mainly about micro measures of Alertness levels in Humans using EEG

Requirements:

EEGlab (tested with eeglab13_5_4b),

fieldtrip (tested with fieldtrip-20151223), 

Matlab (tested with R2016b)

Steps:

If you have a system with 64 electrodes from Neuroscan with the following electrode labels:
Occiptal: 'Oz','O1','O2', Central: 'C3', 'C4', Parietal: 'PO10', Temporal: 'T7','T8','TP8','FT10','TP10', Frontal:'F7', 'F8', 'Fz'
or If you have a system with 128 electrodes from EGI with the following electrode labels:
Occiptal: 'E75','E70','E83', Central: 'E36', 'E104', Parietal:'E90', Temporal:'E45','E108','E102','E114','E100', 
Frontal:'E33', 'E122', 'E11'. 

Then proceed directly to 1) below, if not then look at "Electrode Labelling"

1) Look at the example_demo and add the relevant paths including this toolbox

2) Load your file in the EEGlab format (pretrial epoched data of 4sec duration sampled at 250 Hz)

3) Pass this to the classify_microMeasures function

4) The return values inside the struct trialstruc contains indices of your trials classed as 'Alert', 'Drowsy(mid)', 'Drowsy(grapho)'

   Also has additional details on elements like vertex, spindle, k-complex indices
   
"Electrode Labelling":
If you have a 64 channel system then look at the electrode locations from 64 electrodes from Neuroscan and choose the optimal locations and 
rename your electrode labels to those given in line 14.
If you have a 128 channel system then look at the electrode locations from 128 electrodes from EGI and choose the optimal locations and 
rename your electrode labels to those given in line 16,17.
Once you are done with electrode labelling, go to 1)
