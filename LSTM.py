import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import datasets


#fix random seed for reproducability
numpy.random.seed(7)

#Load Dataset
dataframe = datasets.load_boston()
print(dataframe.data.shape)
dataset = dataframe.data
dataset = dataset.astype('float32')

#normalize
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

trainX = train[0:(len(train)-1),0:11]
trainY = train[1:len(train),12]
testX = test[0:(len(test)-1),0:11]
testY = train[1:len(test),12]

rtrx = numpy.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
rtstx = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))


#create model
model = Sequential()
model.add(LSTM(4,input_shape=(11,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(rtrx,trainY,epochs=3,batch_size=1,verbose=2)

#make predictions
trainPredict = model.predict(rtrx)
testPredict = model.predict(rtstx)

dummy = numpy.zeros(shape=(len(trainPredict), 13) )
dummy[:,0] = trainPredict[:,0]
trainPredict = scaler.inverse_transform(dummy)[:,0]

dummy = numpy.zeros(shape=(len(testPredict), 13) )
dummy[:,0] = testPredict[:,0]
testPredict = scaler.inverse_transform(dummy)[:,0]


dummy = numpy.zeros(shape=(len(trainY),13))	
dummy[:,0] = trainY
trainY = scaler.inverse_transform(dummy)[:,0] 

dummy = numpy.zeros(shape=(len(testY),13))	
dummy[:,0] = testY
testY = scaler.inverse_transform(dummy)[:,0] 

#compile model
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))








