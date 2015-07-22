"""
July 19th, 2015
Kaggle west-nile-virus completition
"""
import pandas as pd
import numpy as np
import pdb
import datetime

def load_weather():
    print("loading weather data ...")
    weather = pd.read_csv('weather.csv', na_values = ['M', '-'])  
    
    # Select only relevant columns
    select_columns = [u'Station', u'Date', u'Tmax', u'Tmin', u'Tavg', u'Depart', u'DewPoint', \
                      u'WetBulb', u'Heat', u'Cool', u'Sunrise', u'Sunset', u'SnowFall', u'PrecipTotal', \
                      u'StnPressure', u'SeaLevel', u'ResultSpeed', u'ResultDir', u'AvgSpeed']
    
    weather = weather[select_columns]
    
    # Add 'Year', 'Month', 'Day' columns
    date_array = np.array(weather.Date.apply(lambda x: x.split('-')).tolist()).astype(int)
    weather[['Year', 'Month', 'Day']] = pd.DataFrame(date_array, columns=['Year', 'Month', 'Day'])
    
    # Fix missing values
    
    # Replace Trace values ('T') for PrecipTotal and SnowFall 
    # with 0.005. 'T' indicates value above 0 but less than 0.01 inches
    weather.loc[weather.PrecipTotal == '  T', 'PrecipTotal'] = 0.005
    weather.loc[weather.SnowFall == '  T', 'SnowFall'] = 0.005
    #pdb.set_trace()
 
 
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
    
    # Assert that weather table consists of two consecutive rows for station 1 and 2
    date_station = np.unique([tuple(v.values.tolist()) for  (k,v) in weather.groupby('Date')['Station']])
    assert (date_station.shape == (1, 2) and date_station[0].tolist() == [1, 2]), \
        "weather table should consists of two consecutive rows for stations 1 and 2"
          
    
    columns = weather.columns.tolist()
    columns = [col for col in columns if not col in ['Station', 'Date', 'Year', 'Month', 'Day']]
    
    for col in columns:
        for row in range(0, weather.shape[0], 2):
            val0 = weather.loc[row, col]
            val1 = weather.loc[row + 1, col]
            if (np.isnan(val0)):
                weather.loc[row, col] = val1
            elif (np.isnan(val1)):
                weather.loc[row + 1, col] = val0

    print("done.")
    return weather[select_columns + ['Year', 'Month', 'Day']]#[['Date', 'Station', 'Tavg']]

def get_day_of_year(date):
    
    ymd = [int(s) for s in date.split('-')]
    date = datetime.date(ymd[0], ymd[1], ymd[2]).isocalendar()[1:3]
    year_first = datetime.date(ymd[0], 1, 1).isocalendar()[1:3]
    day_of_year = 1 + (date[1] - year_first[1]) + (date[0] - year_first[0]) * 7
    return day_of_year, date[0]  
    
def load_train():
    print("loading train data ...")
    train = pd.read_csv('train.csv')#[['Date', 'Trap', 'Latitude', 'Longitude', 'WnvPresent']]

    # Remove unnecessary columns
    columns = [u'Date', u'Species', u'Trap', u'Latitude', u'Longitude', u'AddressAccuracy', u'NumMosquitos', u'WnvPresent']
    train = train[columns]
    
    # Map species to categorical values in range 1-7. The values are sorted by the probability of wnv.
    wnv_prob = train.groupby('Species').WnvPresent.sum()
    species_map = dict(zip(wnv_prob.keys(), range(1, len(wnv_prob) + 1)))
    
    train.Species = train.Species.apply(lambda x: species_map[x])
    
    # Do a similar mapping for traps
    wnv_prob = train.groupby('Trap').WnvPresent.sum()
    trap_map = dict(zip(wnv_prob.keys(), range(1, len(wnv_prob) + 1)))
    train.Trap = train.Trap.apply(lambda x: trap_map[x])
    
    # Add day of year to the table
    train[['Day_of_year', 'Week']] = train.Date.apply(get_day_of_year)         
    print("done.")
    return train


def dist(point1, point2):
    return np.sum((point1 - point2)**2)
    
# Weather stations latitude and longitude
station1 = [41.995, -87.933]
station2 = [41.786, -87.752] 

def find_closest_weather_station(lat, long):
    point1 = np.array([lat, long])
    return np.argmin([dist(point1, np.array(station1)), dist(point1, np.array(station2))]) + 1

def merge_train_weather(train, weather):
     
    weather_columns = weather.columns.tolist()
    weather_columns.remove('Date')
    weather_columns.remove('Station')
    train_columns = train.columns.tolist()
    train_columns.remove('Date')
    
    # Add closest station for each trap
    train['Station'] = [find_closest_weather_station(lat, long) for lat, long in zip(train.Latitude, train.Longitude)]
    # Add weather date for the past and also the following 7 days
    weather_extended = pd.DataFrame()
    for day in (range(-7, 0) + range(1, 8)):
        train['Date_new'] = [(datetime.date(y, m, d) - datetime.timedelta(day)).strftime("20%y-%m-%d") for y, m, d in [map(lambda x : int(x), date.split('-')) for date in train.Date]]
        weather_extended = pd.concat([weather_extended, pd.merge(train, weather, left_on = ['Date_new', 'Station'], right_on = ['Date', 'Station'])[weather_columns]], axis = 1)
    
    pdb.set_trace()
    
    train.drop('Date_new', axis = 1, inplace=True)
    train_weather = pd.concat([train, weather_extended], axis = 1)
    train_weather.to_pickle('train_weather.pkl')
    return train_weather

        
     

if __name__ == '__main__':
    weather = load_weather()
    train = load_train()
    train_weather = load_train_weather()    
            
