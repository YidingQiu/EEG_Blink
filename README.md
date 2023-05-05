### EEG Blink detector
This project provides a blink detection function for EEG data, utilizing various deep learning models and preprocessing techniques. 

Currently, the available functionality is in a function [detectBlink](Package/example.pdf). You just need to input Fp1, Fp2, and Fz (or at least one of Fp1 and Fp2) along with the sampling frequency.

To use the detectBlinks function, ensure that the Package and Models folders are included in your MATLAB search path.


![this is a blink finding function](https://user-images.githubusercontent.com/70067693/233838531-fb55a615-9206-460e-a13d-1cd7e9d054cd.png)

An example of the function is in Package\/[examlpe.pdf](Package/example.pdf) (Package\/[examlpe.mlx](Package/example.mlx)). 

This project is being closely watched and updated continuously. If you have any suggestions or find any issues, please leave your review or discussion, or contact the [email](#contact_info) below.

### File exchange repository
[![View EEG_Blink on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://au.mathworks.com/matlabcentral/fileexchange/120873-eeg_blink)

### Project
The projecs experiments with various deep learning models and EEG preprocessing for blink detection.
Currently the following classes of models have been invistigated:
* LSTM 
* biLSTM
* 1 dimension CNN
    Apply 1 dimentional kernel on time series data.
* TCN 
    Temporal convolutional network. Dilated convolution applied, to increase the receptive field.
* WTCNN     
    3 EEG channels converted to images representing wavelet coefficients, then CNN applied.

Not efficient and discarded models:
* ~WTLSTM~
* ~LSTM followed CNN~

Upcoming experiment on this:
* transformer model applies attention mechanisms

### Experiment pipeline
![image](https://user-images.githubusercontent.com/70067693/236393337-0d293251-c68f-4c1c-aeff-a3ea1241f514.png)
An example of working pipeline rebuilt in beginToEndPipeline.m file. If you want to run it, please download whole EEG_Blink to include all helper function.



### Contributors:
* Yiding Qiu
* Artem Lenskiy


### Contact <a id="contact_info"></a>
Yiding.Qiu@anu.edu.au

qiu-yiding@qq.com

-----
### Dataset


[1]Alexander P. Rockhill and Nicko Jackson and Jobi George and Adam Aron and Nicole C. Swann (2020). UC San Diego Resting State EEG Data from Patients with Parkinson's Disease. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds002778.v1.0.4
https://openneuro.org/datasets/ds002778/versions/1.0.2

[2]Paprocki, Rafal, Gebrehiwot, Temesgen, Gradinscak, Marija, and Artem Lenskiy. "Extracting Blink Rate Variability from EEG Signals." arXiv, (2016). https://doi.org/10.48550/arXiv.1603.03031.


### Dependencies
[1]Deep Learning Toolbox
