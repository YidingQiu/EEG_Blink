### EEG Blink detector
This project provides a blink detection function for EEG data, utilizing various deep learning models and preprocessing techniques. To use the detectBlinks function, ensure that the Package and Models folders are included in your MATLAB search path.

![this is a blink finding function](https://user-images.githubusercontent.com/70067693/233838531-fb55a615-9206-460e-a13d-1cd7e9d054cd.png)


An example of the function is in [Package\examlpe.pdf](Package/example.pdf) ([Package\examlpe.mlx](Package/example.mlx)). 

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





### Contributors:
* Yiding Qiu
* Artem Lenskiy


### Contact
Yiding.Qiu@anu.edu.au
qiu-yiding@qq.com

-----
### Dataset


[1]Alexander P. Rockhill and Nicko Jackson and Jobi George and Adam Aron and Nicole C. Swann (2020). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds002778.v1.0.4
https://openneuro.org/datasets/ds002778/versions/1.0.2

[2]Paprocki, Rafal, Gebrehiwot, Temesgen, Gradinscak, Marija, and Artem Lenskiy. "Extracting Blink Rate Variability from EEG Signals." arXiv, (2016). https://doi.org/10.48550/arXiv.1603.03031.
