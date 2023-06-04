function [OUTEEG,response] = pop_EEG_Blink(EEG)%[,response] =
OUTEEG = EEG;
response = sprintf("EEG_Blink running");
disp(response);
if_plot = true;

%% get interested channel and to table
% eeg_data = EEG.data;
samplingRate = EEG.srate;
% index = struct2table(EEG.chanlocs).labels;
[ch_table, finded_ch] = find_ch(EEG, {'Fp1','Fp2','Fz'});

variables = ch_table.Properties.VariableNames;
for i = 1:length(variables)
    if isa(ch_table.(variables{i}), 'single')
        ch_table.(variables{i}) = double(ch_table.(variables{i}));
    end
end

%% direct call detectBlink
% blink_location = detectBlinks(ch_table, samplingRate, plot = if_plot);
% 
% save("blink_location.mat", "blink_location", '-mat');
% response = sprintf("blink finding done");
%disp(response);
%% pop up window simpleDisplay
appOptions = detectOptions();
blinkDetectorObj = blinkDetector(ch_table, samplingRate, 'TCNOC', appOptions);
simpleDisplay(blinkDetectorObj);




