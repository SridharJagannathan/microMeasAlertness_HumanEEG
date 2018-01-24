The project is mainly about micro measures of Alertness levels in Humans using EEG

Requirements:

EEGlab (tested with eeglab13_5_4b),

fieldtrip (tested with fieldtrip-20151223), 

Matlab (tested with R2016b)

Steps:

1) Look at the example_demo and add the relevant paths including this toolbox

2) Load your file in the EEGlab format (pretrial epoched data of 4sec duration sampled at 250 Hz)

3) Pass this to the classify_microMeasures function

4) The return values inside the struct trialstruc contains indices of your trials classed as 'Alert', 'Drowsy(mid)', 'Drowsy(grapho)'

   Also has additional details on elements like vertex, spindle, k-complex indices