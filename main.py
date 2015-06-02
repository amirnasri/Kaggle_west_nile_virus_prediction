import numpy as np
import csv
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

def load_data(filename, col_parsers = None):
    
    f = open(filename, 'r')
    
    cr = csv.reader(f, delimiter = ',')
    cr.next()
    
    data = []
    for cols in cr:
        if (len(cols) != len(col_parsers)):
            continue
        row = dict()
        for col, col_parser in zip(cols, col_parsers):
            col_name = col_parser[0]
            parser = col_parser[1]
            
            row[col_name] = parser(col) if (col != 'M') else None
            
        data.append(row)
    
    f.close()
        
    return data

def load_weather_data():
    col_parsers = [('Station', int), ('Date', str), ('Tmax', int), ('Tmin', int),
                   ('Tavg', int), ('Depart', int), ('DewPoint', int), ('WetBulb', str), ('Heat', str),
                   ('Cool', str), ('Sunrise', str), ('Sunset', str), ('CodeSum', str), ('Depth', str),
                   ('Water1', str), ('SnowFall', str), ('PrecipTotal', str), ('StnPressure', float),
                   ('SeaLevel', float), ('ResultSpeed', float), ('ResultDir', str), ('AvgSpeed', float)]

    
    data = load_data('weather.csv', col_parsers)
    
    weather_data = {}
    for row in data:
        station = row.pop('Station') - 1
        date = row.pop('Date')
        if date not in weather_data: 
            weather_data[date] = [None, None]
        if weather_data[date][station] is not None:
            print "duplicate weather reading for row {0}".format(row)
        weather_data[date][station] = row
        
    return weather_data
        
stations = np.array([[41.995, -87.933],
                      [41.786, -87.752]], dtype = float)
def find_closest_weather_station(lat, lang):
    loc = np.array([lat, lang])
    return np.argmin(np.sum((loc-stations)**2, axis = 1))
    
    
species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000", 
              'CULEX PIPIENS'   : "001000", 
              'CULEX PIPIENS/RESTUANS' : "101000", 
              'CULEX ERRATICUS' : "000100", 
              'CULEX SALINARIUS': "000010", 
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001000"} # Treating unspecified as PIPIENS (http://www.ajtmh.org/content/80/2/268.full)

def extract_X_y(data):
    X = []
    y = []
    for row in data:
        X_row = []
        if 'NumMosquitos' in row.keys():
            X_row.append(row['NumMosquitos'])
        X_row.extend([float(x) for x in species_map[row['Species']]])
        lang = row['Longitude']
        lat = row['Latitude']
        station = find_closest_weather_station(lat, lang)
        date = row['Date']
        X_row.extend([weather_data[date][station][attr] for attr in ['Tmax', 'Tmin', 'Depart', 'DewPoint', 'StnPressure']])
        X.append(X_row)
        if 'WnvPresent' in row.keys():
            y.append(row['WnvPresent'])
    imp = Imputer(strategy = 'mean', axis = 0)
    return np.array(imp.fit_transform(X)), np.array(y)
    
def load_train_data():
    col_parsers = [("Date", str), ("Address", str), ("Species", str), ("Block", int),
                      ("Street", str), ("Trap", str), ("AddressNumberAndStreet", str),
                      ("Latitude", float), ("Longitude", float),
                     ("AddressAccuracy", float), ("NumMosquitos", int), ("WnvPresent", int)]
    
    data = load_data('train.csv', col_parsers)
    return extract_X_y(data)

def load_test_data():
    col_parsers = [("Id", int), ("Date", str), ("Address", str), ("Species", str), ("Block", int),
                      ("Street", str), ("Trap", str), ("AddressNumberAndStreet", str),
                      ("Latitude", float), ("Longitude", float), ("AddressAccuracy", float)]
    
    data = load_data('test.csv', col_parsers)
    return extract_X_y(data)

if (__name__ == '__main__'):
    
    weather_data = load_weather_data()
    X, y = load_train_data()
    #X_test, y_test = load_test_data()
    X, y = shuffle(X, y, random_state=123)
    m = X.shape[0]
    X_train, y_train = X[:.8 *m ,:], y[:.8 * m]
    X_test, y_test = X[.8 *m: ,:], y[.8 * m:]

    
    clf = LogisticRegression(C = 100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict_proba(X_train)[:, 1]
    print(roc_auc_score(y_train, y_pred))
    print y_train
    print y_pred
    