from __future__ import print_function
import numpy as np
import datetime
import csv
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split
from sys import exit
from IPython.core.debugger import Tracer

species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000", 
              'CULEX PIPIENS'   : "001000", 
              'CULEX PIPIENS/RESTUANS' : "101000", 
              'CULEX ERRATICUS' : "000100", 
              'CULEX SALINARIUS': "000010", 
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001000"} # Treating unspecified as PIPIENS (http://www.ajtmh.org/content/80/2/268.full)

def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()
    
def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "T":
        return TRACE
    return float(text)

def impute_missing_weather_station_values(weather):
    # Stupid simple
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]
    
def load_weather():
    weather = {}
    for line in csv.DictReader(open("weather.csv")):
        for name, converter in {"Date" : date,
                                "Tmax" : float,"Tmin" : float,"Tavg" : float,
                                "DewPoint" : float, "WetBulb" : float,
                                "PrecipTotal" : precip,
                                "Depart" : float, 
                                "ResultSpeed" : float,"ResultDir" : float,"AvgSpeed" : float,
                                "StnPressure" : float, "SeaLevel" : float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
        station = int(line["Station"]) - 1
        assert station in [0,1]
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
        assert weather[dt][station] is None, "duplicate weather reading {0}:{1}".format(dt, station)
        weather[dt][station] = line
    impute_missing_weather_station_values(weather)        
    return weather
    
    
def load_training():
    training = []
    for line in csv.DictReader(open("train.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : float, "Longitude" : float,
                                "NumMosquitos" : int, "WnvPresent" : int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training
    
def load_testing():
    training = []
    for line in csv.DictReader(open("test.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : float, "Longitude" : float}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training
    
    
def closest_station(lat, long):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, long])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1)
    return np.argmin(dist2)
       
def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std
    
def scaled_count(record):
    SCALE = 10.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))
    
    
def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        lat, long = b["Latitude"], b["Longitude"]
        #case = [date.year, date.month, date.day, lat, long]
        case = []
        # Look at a selection of past weather values
        for days_ago in range(1, 8): #[1,3,7,14]
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax","Tmin","Tavg","DewPoint","WetBulb","PrecipTotal","Depart"]:
                station = closest_station(lat, long)
                case.append(weather[day][station][obs])
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        # Weight each observation by the number of mosquitos seen. Test data
        # Doesn't have this column, so in that case use 1. This accidentally
        # Takes into account multiple entries that result from >50 mosquitos
        # on one day. 
        X.append(case)    
    X = np.asarray(X, dtype=np.float32)
    return X
    
def assemble_y(base):
    y = []
    for b in base:
        y.append(b["WnvPresent"])    
    return np.asarray(y, dtype=np.int32).reshape(-1,1)


class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))

def train():
    weather = load_weather()
    training = load_training()
    
    X = assemble_X(training, weather)
    mean, std = normalize(X)
    y = assemble_y(training)
    '''    
    input_size = len(X[0])
    
    learning_rate = theano.shared(np.float32(0.1))
    
    net = NeuralNet(
    layers=[  
        ('input', InputLayer),
         ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, input_size), 
    hidden1_num_units=256, 
    dropout1_p=0.4,
    hidden2_num_units=256, 
    dropout2_p=0.4,
    output_nonlinearity=sigmoid, 
    output_num_units=1, 

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=learning_rate,
    update_momentum=0.9,
    
    # Decay the learning rate
    on_epoch_finished=[
            AdjustVariable(learning_rate, target=0, half_life=4),
            ],

    # This is silly, but we don't want a stratified K-Fold here
    # To compensate we need to pass in the y_tensor_type and the loss.
    regression=True,
    y_tensor_type = T.imatrix,
    objective_loss_function = binary_crossentropy,
     
    max_epochs=32, 
    eval_size=0.1,
    verbose=1,
    )
    '''
    clf = LogisticRegression(C = 10)
    X, y = shuffle(X, y, random_state=123)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)
    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)[:,1]
    print("ROC score", metrics.roc_auc_score(np.ravel(y_test), probas))
    
    clf.fit(X, y)
    return clf, mean, std     
    

def submit(clf, mean, std):
    weather = load_weather()
    testing = load_testing()
    X = assemble_X(testing, weather) 
    normalize(X, mean, std)
    predictions = clf.predict_proba(X)[:,1]
    #Tracer()()    
    #
    out = csv.writer(open("west_nile.csv", "w"))
    out.writerow(["Id","WnvPresent"])
    for row, p in zip(testing, predictions):
        out.writerow([row["Id"], p])

    out = csv.writer(open("west_nile_binary.csv", "w"))
    out.writerow(["Id","WnvPresent"])
    for row, p in zip(testing, (predictions < 0.5)):
        out.writerow([row["Id"], p])


if __name__ == "__main__":
    clf, mean, std = train()
    submit(clf, mean, std)



    