function [signal] = passbandBlinks(signal, fs, filter)
    arguments
        signal  =   [];
        fs      =   512;           % sampling rate
        filter  =   true
    end
    bpm     =   [20 240];      % blinks per minute [min max]
    bps         = bpm / 60;       % blinks per second
    lf_cutoff   = bps(1); % set low cut off frequency
    hf_cutoff   = bps(2); % set hight cut off frequency    

    % remove a linear trend in order to make the first and last values zeros and hence prevent Gibbs phenomenon
    for k = 1:size(signal,2)
        p       = polyfit([1, height(signal)], [signal(1, k).Variables, signal(end, k).Variables], 1); 
        signal(:, k).Variables  = signal(:, k).Variables - (p(1) * [1:height(signal)]' + p(2)); 
    end
    if filter
        signal.Variables = highpass(signal.Variables, lf_cutoff, fs);%, "ImpulseResponse", "iir", "Steepness", 0.85,"StopbandAttenuation",60
        signal.Variables = lowpass (signal.Variables, hf_cutoff, fs);%, "ImpulseResponse", "iir", "Steepness", 0.85,"StopbandAttenuation",60
    end
    
end

