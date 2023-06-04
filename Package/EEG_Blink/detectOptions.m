classdef detectOptions
    properties
        shiftFactor = 3
        tolerance = 10
        plot = 0
        output = {'location'}
        shortSignal = 0
    end
    methods
        function obj = detectOptions(shiftFactor, tolerance, plot, output, shortSignal)
            if nargin > 0
                obj.shiftFactor = shiftFactor;
                obj.tolerance = tolerance;
                obj.plot = plot;
                obj.output = output;
                obj.shortSignal = shortSignal;
            end
        end
    end
end
