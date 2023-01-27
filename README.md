### EEG Blink detector
To use the Blink detector function: detectBlinks, folder Package and Models should include in the MATLAB search path.
An example of the function is in [Package\examlpe.mlx](Package/example.mlx). 

### File exchange repository
[![View EEG_Blink on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://au.mathworks.com/matlabcentral/fileexchange/120873-eeg_blink)

### Project
The projecs experiments with various deep learning models and EEG preprocessing for blink detection.
Currently the following classes of models have been invistigated:
* LSTM 
* biLSTM
* 1 dimension CNN
* TCN 
* CNN 3 EEG channels converted to images representing wavelet coefficients 

Not efficient and discarded models:
* ~LSTM 3 EEG channels converted to images representing wavelet coefficients~
* ~LSTM followed CNN~

### Experiment pipeline
Working pipeling rebuilt in beginToEndPipeline.m file. Download whole EEG_Blink(around 10MB) to include all helper function.
Since big files can not uploaded to GitHub, the data and label is here: [EEG_Blink.zip](https://drive.google.com/file/d/1c0lXKpm8dkcC-b6dH14a1wcKis4JCMuC/view?usp=sharing). Unzip it to root dirctory together with beginToEndPipeline.m.





### Contributors:
* Yiding Qiu
* Artem Lenskiy


### Contact
[email](Yiding.Qiu@anu.edu.au)
[email](qiu-yiding@qq.com)

-----
### Dataset


[1]Alexander P. Rockhill and Nicko Jackson and Jobi George and Adam Aron and Nicole C. Swann (2020). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds002778.v1.0.4
https://openneuro.org/datasets/ds002778/versions/1.0.2

[2]Paprocki, Rafal, Gebrehiwot, Temesgen, Gradinscak, Marija, and Artem Lenskiy. "Extracting Blink Rate Variability from EEG Signals." arXiv, (2016). https://doi.org/10.48550/arXiv.1603.03031.
