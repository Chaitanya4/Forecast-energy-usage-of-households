from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#loading given dataset
result=pd.read_csv("./dataset/Power-Networks-LCL.csv",header=0)
#fetching unique households id
print(result['LCLid'].unique())
dict={}
for i in result['LCLid'].unique():
    dict.update({i:result['LCLid'][result.LCLid == i].count()})
print(dict)
count_list=list(dict.values())
print(count_list)
#function to getting top 3 households
def findmaxelements(initiallist): 
    output_list = [] 
    for i in range(0, 3):  
        maxelement1 = 0  
        for j in range(len(initiallist)):      
            if initiallist[j] > maxelement1: 
                maxelement1 = initiallist[j]          
        initiallist.remove(maxelement1); 
        output_list.append(maxelement1)      
    return output_list
maxelement_list=findmaxelements(count_list)
print(maxelement_list)
print(count_list)
household_list=[]
for j in dict.keys():
    if dict.get(j) in maxelement_list:
        household_list.append(j)
#list storing top 3 households
print(household_list)
#dataframe containing data of household having maximmum no. of samples
new_data1=result[result.LCLid==household_list[0]]
print(new_data1)
#storing dataframe data into preprocessed1
new_csv1=new_data1.to_csv(r'./preprocess-generated-dataset/preprocessed1.csv', index = None, header=True)
resample_data1=pd.read_csv("./preprocess-generated-dataset/preprocessed1.csv",header=0)
resample_data1['date']=pd.to_datetime(resample_data1['DateTime'])
resample_data1=resample_data1.set_index(resample_data1.date)
#resampling data to convert half-hourly data into hourly data ,i.e., downsampling
resample_dataf1=resample_data1.resample('60T').agg({'LCLid':'first','stdorToU':'first', 'DateTime':'first','KWh': np.mean, 'Acorn':'first','Acorn_grouped':'first'})
#storing resampled data of household having maximmum no. of samples in resampled1
resample_csv1=resample_dataf1.to_csv(r'./preprocess-generated-dataset/resampled1.csv', index = None, header=True)
print(resample_dataf1)
#visualizing resampled data of household having maximmum no. of samples
resample_dataf1.plot(style=[':','--','-'])
#dataframe containing data of household having 2nd maximmum no. of samples
new_data2=result[result.LCLid==household_list[1]]
print(new_data2)
#storing dataframe data into preprocessed2
new_csv2=new_data2.to_csv(r'./preprocess-generated-dataset/preprocessed2.csv', index = None, header=True)
resample_data2=pd.read_csv("./preprocess-generated-dataset/preprocessed2.csv",header=0)
resample_data2['date']=pd.to_datetime(resample_data2['DateTime'])
resample_data2=resample_data2.set_index(resample_data2.date)
#resampling data to convert half-hourly data into hourly data ,i.e., downsampling
resample_dataf2=resample_data2.resample('60T').agg({'LCLid':'first','stdorToU':'first', 'DateTime':'first','KWh': np.mean, 'Acorn':'first','Acorn_grouped':'first'})
#storing resampled data of household having 2nd maximmum no. of samples in resampled2
resample_csv2=resample_dataf2.to_csv(r'./preprocess-generated-dataset/resampled2.csv', index = None, header=True)
print(resample_dataf2)
#visualizing resampled data of household having 2nd maximmum no. of samples
resample_dataf2.plot(style=[':','--','-'])
#dataframe containing data of household having 3rd maximmum no. of samples
new_data3=result[result.LCLid==household_list[2]]
print(new_data3)
#storing dataframe data into preprocessed3
new_csv3=new_data3.to_csv(r'./preprocess-generated-dataset/preprocessed3.csv', index = None, header=True)
resample_data3=pd.read_csv("./preprocess-generated-dataset/preprocessed3.csv",header=0)
resample_data3['date']=pd.to_datetime(resample_data3['DateTime'])
resample_data3=resample_data3.set_index(resample_data3.date)
#resampling data to convert half-hourly data into hourly data ,i.e., downsampling
resample_dataf3=resample_data3.resample('60T').agg({'LCLid':'first','stdorToU':'first', 'DateTime':'first','KWh': np.mean, 'Acorn':'first','Acorn_grouped':'first'})
#storing resampled data of household having 3rd maximmum no. of samples in resampled2
resample_csv3=resample_dataf3.to_csv(r'./preprocess-generated-dataset/resampled3.csv', index = None, header=True)
print(resample_dataf3)
#visualizing resampled data of household having 3rd maximmum no. of samples
resample_dataf3.plot(style=[':','--','-'])


#creating model for household MAC000018
final_data1=resample_dataf1.loc[:,['KWh']]
final_data1=final_data1.set_index(resample_dataf1.DateTime)
print(final_data1.head())
values=final_data1['KWh'].values.reshape(-1,1)
values=values.astype('float32')
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)
train_size=int(len(scaled)*0.8)
test_size=len(scaled)-train_size
train,test=scaled[0:train_size,:],scaled[train_size:len(scaled),:]
print(len(train),len(test))
def create_dataset(dataset,look_back=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-look_back):
        a=dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    print(len(dataY))
    return np.array(dataX),np.array(dataY)
look_back=2
trainX,trainY=create_dataset(train,look_back)
testX,testY=create_dataset(test,look_back)
trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX=np.reshape(testX,(testX.shape[0],1,testX.shape[1]))
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model = Sequential()
model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history = model.fit(trainX, trainY, epochs=30, batch_size=400, validation_data=(testX, testY), verbose=1, shuffle=False)
model.summary()
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# invert predictions
train_predict = scaler.inverse_transform(train_predict)
trainY = scaler.inverse_transform([trainY])
test_predict = scaler.inverse_transform(test_predict)
testY = scaler.inverse_transform([testY])
aa=[x for x in range(1600)]
plt.figure(figsize=(8,4))
plt.plot(aa, testY[0][:1600], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:1600], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Energy Consumption', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.title("Energy usage in kwh for household 'MAC000018'", fontsize=14, fontstyle='italic', fontweight='bold')
#checking model performance by using RMSE & MAE
print('Train Mean Absolute Error:', mean_absolute_error(trainY[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(trainY[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(testY[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(testY[0], test_predict[:,0])))



#creating model for household MAC000020
final_data2=resample_dataf2.loc[:,['KWh']]
final_data2=final_data2.set_index(resample_dataf2.DateTime)
print(final_data2.head())
values2=final_data2['KWh'].values.reshape(-1,1)
values2=values2.astype('float32')
scaler2=MinMaxScaler(feature_range=(0,1))
scaled2=scaler2.fit_transform(values2)
train_size2=int(len(scaled2)*0.8)
test_size2=len(scaled2)-train_size2
train2,test2=scaled2[0:train_size2,:],scaled2[train_size2:len(scaled2),:]
print(len(train2),len(test2))
trainX2,trainY2=create_dataset(train2,look_back)
testX2,testY2=create_dataset(test2,look_back)
trainX2=np.reshape(trainX2,(trainX2.shape[0],1,trainX2.shape[1]))
testX2=np.reshape(testX2,(testX2.shape[0],1,testX2.shape[1]))
print(trainX2.shape, trainY2.shape, testX2.shape, testY2.shape)
model = Sequential()
model.add(LSTM(100, input_shape=(trainX2.shape[1], trainX2.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history2 = model.fit(trainX2, trainY2, epochs=30, batch_size=400, validation_data=(testX2, testY2), verbose=1, shuffle=False)
model.summary()
train_predict2 = model.predict(trainX2)
test_predict2 = model.predict(testX2)
# invert predictions
train_predict2 = scaler2.inverse_transform(train_predict2)
trainY2 = scaler2.inverse_transform([trainY2])
test_predict2 = scaler2.inverse_transform(test_predict2)
testY2 = scaler2.inverse_transform([testY2])
aa2=[x2 for x2 in range(1600)]
plt.figure(figsize=(8,4))
plt.plot(aa2, testY2[0][:1600], marker='.', label="actual")
plt.plot(aa2, test_predict2[:,0][:1600], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Energy Consumption', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.title("Energy usage in kwh for household 'MAC000020'", fontsize=14, fontstyle='italic', fontweight='bold')

#checking model performance by using RMSE & MAE
print('Train Mean Absolute Error:', mean_absolute_error(trainY2[0], train_predict2[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(trainY2[0], train_predict2[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(testY2[0], test_predict2[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(testY2[0], test_predict2[:,0])))



#creating model for household MAC000021
final_data3=resample_dataf3.loc[:,['KWh']]
final_data3=final_data3.set_index(resample_dataf3.DateTime)
print(final_data3.head())
values3=final_data3['KWh'].values.reshape(-1,1)
values3=values3.astype('float32')
scaler3=MinMaxScaler(feature_range=(0,1))
scaled3=scaler3.fit_transform(values3)
train_size3=int(len(scaled3)*0.8)
test_size3=len(scaled3)-train_size3
train3,test3=scaled3[0:train_size3,:],scaled3[train_size3:len(scaled3),:]
print(len(train3),len(test3))
trainX3,trainY3=create_dataset(train3,look_back)
testX3,testY3=create_dataset(test3,look_back)
trainX3=np.reshape(trainX3,(trainX3.shape[0],1,trainX3.shape[1]))
testX3=np.reshape(testX3,(testX3.shape[0],1,testX3.shape[1]))
print(trainX3.shape, trainY3.shape, testX3.shape, testY3.shape)
model = Sequential()
model.add(LSTM(100, input_shape=(trainX3.shape[1], trainX3.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
history3 = model.fit(trainX3, trainY3, epochs=30, batch_size=400, validation_data=(testX3, testY3), verbose=1, shuffle=False)
model.summary()
train_predict3 = model.predict(trainX3)
test_predict3 = model.predict(testX3)
# invert predictions
train_predict3 = scaler3.inverse_transform(train_predict3)
trainY3 = scaler3.inverse_transform([trainY3])
test_predict3 = scaler3.inverse_transform(test_predict3)
testY3 = scaler3.inverse_transform([testY3])
aa3=[x3 for x3 in range(1600)]
plt.figure(figsize=(8,4))
plt.plot(aa3, testY3[0][:1600], marker='.', label="actual")
plt.plot(aa3, test_predict3[:,0][:1600], 'r', label="prediction")
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Energy Consumption', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.title("Energy usage in kwh for household 'MAC000021'", fontsize=14, fontstyle='italic', fontweight='bold')
plt.show()
#checking model performance by using RMSE & MAE
print('Train Mean Absolute Error:', mean_absolute_error(trainY3[0], train_predict3[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(trainY3[0], train_predict3[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(testY3[0], test_predict3[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(testY3[0], test_predict3[:,0])))
