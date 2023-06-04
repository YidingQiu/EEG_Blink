function [data_interested, channels_found] = find_ch(EEG, ch_list)
    % Convert the channel labels to a table
    chanlocs_table = struct2table(EEG.chanlocs);
    
    % Initialize an empty table for the interested data
    data_interested = table;
    channels_found = {};
    
    % Iterate over each channel in the provided list
    for i = 1:length(ch_list)
        % Find the index of the channel in the EEG data
        ch_idx = find(strcmp(chanlocs_table.labels, ch_list{i}));
        
        % If the channel was found, append its data to the output
        if ~isempty(ch_idx)
            data_interested.(matlab.lang.makeValidName(ch_list{i})) = EEG.data(ch_idx, :)';
            channels_found = [channels_found, ch_list{i}];
        else
            warning('Channel %s not found in the EEG data.', ch_list{i});
        end
    end
    
    % If no channels were found, return an error
    if isempty(channels_found)
        error('No channels from the provided list were found in the EEG data.');
    end
end
