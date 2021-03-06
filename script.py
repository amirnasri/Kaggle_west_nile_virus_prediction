"""
July 19th, 2015
Kaggle west-nile-virus competition

August 3th:
TODO: 
- Fix nan produced by normalize
- Add prediction based on all years (do similar preparation for test data and make predictions)
- Find optimal parameters for GBC by cross-validation
"""
import pandas as pd
import numpy as np
import pdb
import datetime
import sklearn.ensemble
from sklearn.metrics import roc_auc_score
import csv
import sklearn.linear_model

class Kaggle_WNV():

    @staticmethod
    def load_weather():
        print("loading weather data ...")
        weather = pd.read_csv('weather.csv', na_values = ['M', '-'])  
        
        # Select only relevant columns
        select_columns = [u'Station', u'Date', u'Tmax', u'Tmin', u'Tavg', u'Depart', u'DewPoint', \
                          u'WetBulb', u'Heat', u'Cool', u'Sunrise', u'Sunset', u'SnowFall', u'PrecipTotal', \
                          u'StnPressure', u'SeaLevel', u'ResultSpeed', u'ResultDir', u'AvgSpeed']
        
        weather = weather[select_columns]
        
        # Fix missing values
        
        # Replace Trace values ('T') for PrecipTotal and SnowFall 
        # with 0.005. 'T' indicates value above 0 but less than 0.01 inches
        weather.loc[weather.PrecipTotal == '  T', 'PrecipTotal'] = 0.005
        weather.loc[weather.SnowFall == '  T', 'SnowFall'] = 0.005
     
        # Add 'Year', 'Month', 'Day' columns
        #date_array = np.array(weather.Date.apply(lambda x: x.split('-')).tolist()).astype(int)
        #weather[['Year', 'Month', 'Day']] = pd.DataFrame(date_array, columns=['Year', 'Month', 'Day'])
        #date_array = np.array(weather.Date.apply(lambda x: x.split('-'))[0]).astype(int)
        weather['Year'] = weather.Date.apply(lambda x: x.split('-')[0]).astype(int)
     
        weather_stations = [weather[weather.Station == 1], \
                            weather[weather.Station == 2]]
        
        years = np.unique(weather.Year)
        
        weather_interp = pd.DataFrame()
        for year in years:
            for weather in weather_stations:
                weather_interp = pd.concat([weather_interp, weather[weather.Year == year].interpolate()])
            
            
        weather = weather_interp.sort_index()
        
        weather['Depart'] = weather.Depart.astype(np.float64)
        weather['SnowFall'] = weather.SnowFall.astype(np.float64)
        weather['PrecipTotal'] = weather.PrecipTotal.astype(np.float64)
        
        # For the remaining missing values, get the missing value from the other stations for the same date
        
        # Assert that weather table consists of consecutive rows for station 1 and 2
        date_station = np.unique([tuple(v.values.tolist()) for  (k,v) in weather.groupby('Date')['Station']])
        assert (date_station.shape == (1, 2) and date_station[0].tolist() == [1, 2]), \
            "weather table should consists of two consecutive rows for stations 1 and 2"
              
        
        columns = weather.columns.tolist()
        columns = [col for col in columns if not col in ['Station', 'Date', 'Year']]
        
        '''
        for col in columns:
            for row in range(0, weather.shape[0], 2):
                val0 = weather.loc[row, col]
                val1 = weather.loc[row + 1, col]
                if (np.isnan(val0)):
                    weather.loc[row, col] = val1
                elif (np.isnan(val1)):
                    weather.loc[row + 1, col] = val0
        '''            
        
        weather_st1 = weather[0::2][columns]
        weather_st2 = weather[1::2][columns]
        
        weather_st2_index = weather_st2.index
        weather_st2.index = weather_st1.index 
        weather_st1[pd.isnull(weather_st1)] = weather_st2
        weather_st2[pd.isnull(weather_st2)] = weather_st1
        weather_st2.index = weather_st2_index
        
        weather_impuned = pd.concat([weather_st1, weather_st2]).sort_index()
        weather_impuned = pd.concat([weather_impuned, weather[['Station', 'Date', 'Year']]], axis = 1)
            
        print("done.\n")
        return weather_impuned[select_columns]#[['Date', 'Station', 'Tavg']]
    
    '''
       return year, month, day, day_of_year, weak 
    '''
    @staticmethod
    def get_date_info(date):
        
        ymd = [int(s) for s in date.split('-')]
        date = datetime.date(ymd[0], ymd[1], ymd[2]).isocalendar()[1:3]
        year_first = datetime.date(ymd[0], 1, 1).isocalendar()[1:3]
        day_of_year = 1 + (date[1] - year_first[1]) + (date[0] - year_first[0]) * 7
        
        # year, month, day, day_of_year, weak
        #return pd.Series([ymd[0], ymd[1], ymd[2], day_of_year, date[0]])  
        return [ymd[0], ymd[1], ymd[2], day_of_year, date[0]]  
        
        
    def load_train(self):
        
        print("loading train data ...")
        train = pd.read_csv('train.csv')#[['Date', 'Trap', 'Latitude', 'Longitude', 'WnvPresent']]
        
                
        # Remove unnecessary columns
        columns = [u'Date', u'Species', u'Trap', u'Latitude', u'Longitude', u'AddressAccuracy', u'NumMosquitos', u'WnvPresent']
        train = train[columns]
        
        # Map species to categorical values in range 1-7. The values are sorted by the probability of wnv in decreasing order.
        wnv_prob = train.groupby('Species').WnvPresent.sum()
        species_list = wnv_prob.keys().tolist()
        species_list.append('UNSPECIFIED CULEX')
        self.species_map = dict(zip(species_list, range(1, len(wnv_prob) + 2)))
        # TODO: sorting is not done!
        
        train.Species = train.Species.apply(lambda x: self.species_map[x])
        
        # Do a similar mapping for traps
        wnv_prob = train.groupby('Trap').WnvPresent.sum()
        self.trap_map = dict(zip(wnv_prob.keys(), range(1, len(wnv_prob) + 1)))
        train.Trap = train.Trap.apply(lambda x: self.trap_map[x])
        # TODO: sorting is not done!
    
        # TODO: combine records with same Date, Species, and Trap.
            
        # Add day of year to the table
        #train[['Year', 'Month', 'Day', 'Day_of_year', 'Week']] = pd.DataFrame(train.Date.apply(get_day_of_year))         
        train[['Year', 'Month', 'Day', 'Day_of_year', 'Week']] = pd.DataFrame(np.array(train.Date.apply(Kaggle_WNV.get_date_info).tolist()))         
        
        print("done.\n")
        return train
    
    def map_traps_test(self, trap):
        if trap[-1] in list('ABC'):
            trap = trap[:-1]
        
        if trap == 'T234':
            trap = 'T235' 
        #if trap in self.trap_map:
        return self.trap_map[trap]
            
    def load_test(self):
        
        print("loading test data ...")
        test = pd.read_csv('test.csv')
   
        # Test data does not have the 'NumMosquitos' column. Assume NumMosquitos to be one. 
        test['NumMosquitos'] = 1
        
        # Remove unnecessary columns
        columns = [u'Date', u'Species', u'Trap', u'Latitude', u'Longitude', u'AddressAccuracy', u'NumMosquitos']
        test = test[columns]
        
        test.Species = test.Species.apply(lambda x: self.species_map[x])
        
        test.Trap = test.Trap.apply(self.map_traps_test)
    
        # Add day of year to the table
        test[['Year', 'Month', 'Day', 'Day_of_year', 'Week']] = pd.DataFrame(np.array(test.Date.apply(Kaggle_WNV.get_date_info).tolist()))         
        
        print("done.\n")
        return test
        
    @staticmethod
    def dist(point1, point2):
        return np.sum((point1 - point2)**2)
        
    # Weather stations latitude and longitude
    station1 = [41.995, -87.933]
    station2 = [41.786, -87.752]
     
    @staticmethod
    def find_closest_weather_station(lat, long):
        point1 = np.array([lat, long])
        return np.argmin([Kaggle_WNV.dist(point1, np.array(Kaggle_WNV.station1)), Kaggle_WNV.dist(point1, np.array(Kaggle_WNV.station2))]) + 1
    
    @staticmethod
    def merge_data_weather(trap_data, weather):
         
        weather_columns = weather.columns.tolist()
        weather_columns.remove('Date')
        weather_columns.remove('Station')
        trap_data_columns = trap_data.columns.tolist()
        trap_data_columns.remove('Date')
        
        # Add closest station for each trap
        trap_data['Station'] = [Kaggle_WNV.find_closest_weather_station(lat, long) for lat, long in zip(trap_data.Latitude, trap_data.Longitude)]
        
        # Add weather data for the past and also the following 7 days
        weather_extended = pd.DataFrame()
        for day in (range(-7, 8)):
            trap_data['Date_new'] = [(datetime.date(y, m, d) + datetime.timedelta(day)).strftime("20%y-%m-%d") for y, m, d in [map(lambda x : int(x), date.split('-')) for date in trap_data.Date]]
            weather_extended = pd.concat([weather_extended, pd.merge(trap_data, weather, left_on = ['Date_new', 'Station'], right_on = ['Date', 'Station'])[weather_columns]], axis = 1)
        
        
        trap_data.drop('Date_new', axis = 1, inplace=True)
        trap_data.drop('Date', axis = 1, inplace=True)
        trap_weather = pd.concat([trap_data, weather_extended], axis = 1)
        #trap_weather.to_pickle('trap_weather.pkl')
        return trap_weather
    
    @staticmethod
    def normalize_matrix(X):
        #pdb.set_trace()
        mean_X = np.mean(X, axis = 0)
        std_X = np.std(X, axis = 0)
        std_X[std_X == 0] = 1
        X = X - mean_X
        X = X / std_X
        return X, mean_X, std_X
        
            
    def train_GBC(self, train_weather):
        
        '''
        params = {"n_estimators": 10, "learning_rate": 0.0035, \
                          "loss": "deviance", \
                          "max_features": 2, "max_depth": 2, \
                          "random_state": 788942,  \
                          "subsample": 1, "verbose": 50}
        params = {"n_estimators": 1000, "learning_rate": 0.0035, \
                          "loss": "deviance", \
                          "max_features": 8, "max_depth": 7, \
                          "random_state": 788942,  \
                          "subsample": 1, "verbose": 50}
        
        
        gbc = sklearn.ensemble.GradientBoostingClassifier()
        
        gbc.set_params(**params)
        '''
        gbc = sklearn.linear_model.LogisticRegression(C=.1)
        
        y = np.array(train_weather.WnvPresent).ravel()
        train_weather.drop('WnvPresent', axis = 1, inplace = True)
        X = np.array(train_weather)
        X = Kaggle_WNV.normalize_matrix(X)[0]
        
        
        scores = []
        
        for year in range(2007, 2014, 2):
            train_index = np.array(train_weather.Year != year)
            test_index = np.array(train_weather.Year == year)
            gbc.fit(X[train_index], y[train_index])
            
            y_test_pred = gbc.predict_proba(X[test_index])[:, 1]
        
            score = roc_auc_score(y[test_index], y_test_pred)
            print score
            scores.append(score)
            #pdb.set_trace()
        
        print scores
        
        pdb.set_trace()
        

        print("training gbc ...")
        gbc.fit(X, y)
        self.gbc = gbc
        print("done.\n")

    def predict_GBC(self, test_weather):
        
        X = np.array(test_weather)
        X = Kaggle_WNV.normalize_matrix(X)[0]
        
        y_test_pred = self.gbc.predict_proba(X)[:, 1]
        pdb.set_trace()
        
        out = csv.writer(open('submit_file.csv', 'w'))
        out.writerow(["Id", "WnvPresent"])
        for id, pred_val in zip(range(1, len(y_test_pred) + 1), y_test_pred):
            out.writerow([id, pred_val])
            

if __name__ == '__main__':
    kw = Kaggle_WNV()
    weather = kw.load_weather()
    train = kw.load_train()
    train_weather = kw.merge_data_weather(train, weather)    
    
    kw.train_GBC(train_weather)
    
    test = kw.load_test()
    test_weather = kw.merge_data_weather(test, weather)
    kw.predict_GBC(test_weather)