import numpy as np
import csv

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
         

if (__name__ == '__main__'):
    
    col_parsers = [("Date", str), ("Address", str), ("Species", str), ("Block", int),
                      ("Street", str), ("Trap", str), ("AddressNumberAndStreet", str),
                      ("Latitude", float), ("Longitude", float),
                     ("AddressAccuracy", float), ("NumMosquitos", int), ("WnvPresent", int)]
    
    data = load_data('train.csv', col_parsers)
    
    weather_data = load_weather_data()