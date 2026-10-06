The project is mainly about micro measures of Alertness levels in Humans using EEG

## Requirements:
### Software:
* Matlab (tested with R2020b)
* EEGlab (tested with eeglab13_5_4b)
* fieldtrip (tested with fieldtrip-20151223)
### Data:
* pretrial epoched data 
* minimum of 4sec duration 
* sampled at 250 Hz
* Cz referenced

## Steps:

* If you have a system with 64 electrodes from Neuroscan with the following electrode labels: 
```
Occiptal: 'Oz','O1','O2', Central: 'C3', 'C4', Parietal: 'PO10', Temporal: 'T7','T8','TP8','FT10','TP10', 
Frontal:'F7', 'F8', 'Fz'      
```                                     
* If you have a system with 128 electrodes from EGI with the following electrode labels: 
```
Occiptal: 'E75','E70','E83', Central: 'E36', 'E104', Parietal:'E90', Temporal:'E45','E108','E102','E115','E100',        
Frontal:'E33', 'E122', 'E11'    
```              
* If you have a system with 256 electrodes from EGI with the following electrode labels:    
```
Occiptal: 'E126','E116','E150', Central: 'E59', 'E183', Parietal:'E161', Temporal:'E69','E202','E179','E219','E190',        
Frontal:'E47', 'E2', 'E21'
```
* If you have a system with the montage for sleep scoring (10-20 system) with the following electrode labels (use `channelconfig = 'sleep'` and the model `model_collec_elecsleep`): 
```
Occiptal: 'O1','O2', Central: 'C3', 'C4', Parietal: 'P4', Temporal (proxy): 'P3','Pz', 
Frontal:'F3', 'F4', 'Fz'      
```
  See [demo_lists/example_demo_sleep.m](demo_lists/example_demo_sleep.m) for an example of renaming channel labels to this montage.

Then proceed directly to 1. below, if not then look at the section **Electrode Labelling**

1. Look at the [example_demo](example_demo.m), set the paths to your fieldtrip and EEGlab installations and run it. It classifies the bundled example dataset in [example_data/](example_data/) (64 channel Neuroscan, subject 122 from Jagannathan et al., NeuroImage 2018).
2. Load your file in the EEGlab format
3. Pass this to the [classify_microMeasures](classify_microMeasures.m) function along with the model file and the `channelconfig` (`'64'`, `'128'`, `'256'` or `'sleep'`) as shown in [example_demo](example_demo.m)
4. The return values inside the struct trialstruc contains indices of your trials classed as `alert`, `drowsymild`, `drowsysevere`. 
Also has additional details on elements like vertex (`vertgrapho`), spindle (`spingrapho`), k-complex (`kcompgrapho`) indices

## Models:
* `models/model_collec_elec64.mat` - for the 64, 128 and 256 channel configurations
* `models/model_collec_elecsleep.mat` - for the sleep montage configuration

## LIBSVM:
The toolbox uses the bundled [libsvm-3.12](libsvm-3.12/matlab/). Precompiled MEX files are included for Linux (`.mexa64`), Windows (`libsvm-3.12/windows/*.mexw64`) and macOS Intel (`.mexmaci64`, also works with Intel MATLAB under Rosetta on Apple Silicon). For other platforms (e.g. native Apple Silicon MATLAB, `.mexmaca64`) compile them by running `make` inside `libsvm-3.12/matlab` from MATLAB.

The spindle detection requires the MATLAB Wavelet Toolbox and the Signal Processing Toolbox.
   
## Electrode Labelling:           
* If you have a 64 channel system then look at the electrode locations from 64 electrodes from Neuroscan and choose the optimal locations and rename your electrode labels to those given for Neuroscan.           
* If you have a 128 or 256 channel system then look at the electrode locations from 128 or 256 electrodes from EGI and choose the optimal locations and rename your electrode labels to those given for EGI.     
* Once you are done with electrode labelling, go to 1.
