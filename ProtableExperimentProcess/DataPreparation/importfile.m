function loadRaw = importfile(filename, dataLines, channels)
%IMPORTFILE Import data from a text file
%  HC1 = IMPORTFILE(FILENAME) reads data from text file FILENAME for the
%  default selection.  Returns the data as a table.
%
%  HC1 = IMPORTFILE(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Example:
%  loadRaw = importfile("EEG_Blink\RawData\HC_1.csv", [1, Inf]);
%
%  See also READTABLE.
%
% Auto-generated by MATLAB on 2023-04-18 04:17:24

arguments
    filename
    dataLines = [2, Inf];
    channels = ["Fp1", "Fp2", "Fz"]
end

%% Input handling
dataLines = dataLines+1;
% % If dataLines is not specified, define defaults
% if nargin < 2
%     dataLines = [1, Inf];
% end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 32);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Fp1", "AF3", "F7", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23", "Var24", "Var25", "Var26", "Var27", "F8", "AF4", "Fp2", "Fz", "Var32"];
opts.SelectedVariableNames = channels;
opts.VariableTypes = ["double", "double", "double", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, [  "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23", "Var24", "Var25", "Var26", "Var27", "Var32"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, [  "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10", "Var11", "Var12", "Var13", "Var14", "Var15", "Var16", "Var17", "Var18", "Var19", "Var20", "Var21", "Var22", "Var23", "Var24", "Var25", "Var26", "Var27", "Var32"], "EmptyFieldRule", "auto");

% Import the data
loadRaw = readtable(filename, opts);

end