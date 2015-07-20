"""
July 19th, 2015
Kaggle west-nile-virus completition
"""
import pandas as pd
import numpy as np
import pdb


def load_weather():
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

            
    return weather

if __name__ == '__main__':
    weather = load_weather()
        
            
