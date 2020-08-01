# Forecast-energy-usage-of-households
Project is developed using python libraries numpy, panda, sklearn, seaborn, matplotlib and datetime for the TCS HumAIn 2019 contest for forecasting energy usage of households using LSTM model based on the provided dataset.

Run the project from command-line using: python3 forecast.py

Here, is the video of project demo with explanation : https://youtu.be/f9e0vrtXMTI

Forecast the electricity consumption of top 3 households with highest number of samples on an hourly basis based on the previous usage pattern using Power-Networks-LCL.csv dataset. There are six attributes whttps://youtu.be/f9e0vrtXMTIhich includes household id(i.e., LCLid), standard plans used(i.e., stdorToU), date&time(i.e., DateTime), electricity consumption readings in kWh (per half hour) (i.e., KWh), acorn(i.e., Acorn) and acorn groups(i.e., Acorn_grouped) given in the dataset. Using this dataset, we have to forecast the electricity consumption based on the following approach. First, choose the top 3 households with highest number of samples using LCLid attribute. Then, resample the given dataset(i.e., perform downsampling) so as to convert the given half hourly based dataset into hourly based dataset. After pre-processing the dataset in the above manner create a model to predict the pattern of energy consumption using LSTM(i.e., Long Short-Term Memory recurrent neural network).


