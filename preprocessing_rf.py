import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# So that we may see all columns
pd.set_option('display.max_columns', None)


def preprocess_data():
    # Read in the data
    data = pd.read_csv("weatherAUS.csv")
    
    # Convert location to state
    in_victoria = ['Sale', 'Nhil', 'Watsonia', 'Richmond', 'Portland', 'Dartmoor',
                    'Ballarat', 'Bendigo', 'MelbourneAirport', 'Mildura', 'Melbourne']
    in_nsw = ['BadgerysCreek', 'WaggaWagga', 'Wollongong', 'Moree', 'SydneyAirport',
               'Sydney', 'Albury', 'Penrith', 'CoffsHarbour', 'Williamtown', 'Cobar',
               'NorahHead', 'Newcastle']
    in_queensland = ['Townsville', 'GoldCoast', 'Brisbane', 'Cairns']
    in_northern_territory = ['Uluru', 'AliceSprings', 'Katherine', 'Darwin']
    in_south_australia = ['Nuriootpa', 'MountGambier', 'Woomera', 'Adelaide']
    in_west_australia = ['Albany', 'Perth', 'PearceRAAF', 'Walpole', 'SalmonGums', 
                          'PerthAirport', 'Witchcliffe']
    in_capital = ['Tuggeranong', 'MountGinini', 'Canberra']
    in_other = ['Launceston', 'Hobart', 'NorfolkIsland']
    
    data["State"] = 0
    data["State"] = data["Location"].apply(lambda a : "Victoria" if a in in_victoria else a)
    data["State"] = data["State"].apply(lambda a : "New South Wales" if a in in_nsw else a)
    data["State"] = data["State"].apply(lambda a : "Queensland" if a in in_queensland else a)
    data["State"] = data["State"].apply(lambda a : "Northern Territory" if a in in_northern_territory else a)
    data["State"] = data["State"].apply(lambda a : "South Australia" if a in in_south_australia else a)
    data["State"] = data["State"].apply(lambda a : "West Australia" if a in in_west_australia else a)
    data["State"] = data["State"].apply(lambda a : "Capital Territory" if a in in_capital else a)
    data["State"] = data["State"].apply(lambda a : "Other" if a in in_other else a)
    
    # Save results to add back in later but so they don't get messed up by standardization
    state = data['State']
    data = data.drop('State', axis=1)
    
    # Calculate dew point and dew point diff (from min temp)
    # Formula:
    # ((RH/100)^(1/8)*(112+0.9*AT) )+(0.1*AT)-112
    def calc_dew_point(humid, temp):
        humid = humid.apply(lambda a : math.pow(a/100, 1/8))
        humid = humid * (112 + 0.9*temp)
        dew = humid + (0.1 * temp) - 112
        return(dew)
    
    data['DewPoint3pm'] = calc_dew_point(data['Humidity3pm'], data['Temp3pm'])
    data['DewPoint3pmDiff'] = data['Temp3pm'] - data['DewPoint3pm']
    data['DewPoint9am'] = calc_dew_point(data['Humidity9am'], data['Temp9am'])
    data['DewPoint9amDiff'] = data['Temp9am'] - data['DewPoint9am']
    
    # Convert 9pm/3pm data values to "delta" values and drop the originals from the dataframe
    data["deltaWindSpeed"] = data["WindSpeed3pm"] - data["WindSpeed9am"]
    data["deltaHumidity"] = data["Humidity3pm"] - data["Humidity9am"]
    data["deltaPressure"] = data["Pressure3pm"] - data["Pressure9am"]
    data["deltaCloud"] = data["Cloud3pm"] - data["Cloud9am"]
    data["deltaTemp"] = data["Temp3pm"] - data["Temp9am"]
    data["RainToday"] = np.where(data["RainToday"] == "Yes", 1, 0)
    data["RainTomorrow"] = np.where(data["RainTomorrow"] == "Yes", 1, 0)
    data = data.drop(
        ["Location"], axis=1)
    # data = data.drop(
    #     ["Location", "WindSpeed9am", "Humidity9am", "Pressure9am",
    #      "Cloud9am", "Temp9am"], axis=1)
    # data = data.drop(
    #     ["Location", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    #      "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"], axis=1)

    # Filling in NA values
    data = data.interpolate()
    
    # Multivarite imputation (would need to convert everything to numeric first)
    # imp = IterativeImputer(max_iter=10, random_state=0)
    # imp.fit(data)
    # data = np.round(imp.transform(data))

    # http: // www.land - navigation.com / boxing - the - compass.html
    d = {'N': 0, 'NNE': 1, 'NE': 2, 'ENE': 3, 'E': 4, 'ESE': 5, 'SE': 6, 'SSE': 7, 'S': 8, 'SSW': 9, 'SW': 10,
         'WSW': 11, 'W': 12, 'WNW': 13, 'NW': 14, 'NNW': 15}

    data['WindGustDir'] = data['WindGustDir'].map(d)
    data['WindDir9am'] = data['WindDir9am'].map(d)
    data['WindDir3pm'] = data['WindDir3pm'].map(d)

    data['WindDirDelta'] = (data['WindDir3pm'] - data['WindDir9am']) % 16

    # Remove rows with NA values from the data
    dataNoNA = data.dropna()

    dateData = pd.to_datetime(dataNoNA['Date'])
    dataNoNA = dataNoNA.drop(["Date"], axis=1)

    # Separate labels from the data
    labels = dataNoNA["RainTomorrow"]
    dataNoNA = dataNoNA.drop(["RainTomorrow"], axis=1)

    # Standardization
    rain_today = dataNoNA['RainToday']
    dataNoNA = (dataNoNA - dataNoNA.mean()) / dataNoNA.std()
    dataNoNA['RainToday'] = rain_today
    
    # Dummy variables (for state)
    dataNoNA['State'] = state
    dataNoNA = pd.get_dummies(dataNoNA, columns=['State'], drop_first=True)
    
    dataNoNA["Date"] = dateData
    
    # Splitting date into day, month year, and converting to integer
    def split_date(df):
        split_date = df
        split_date['Date'] = split_date['Date'].astype(str)
        split_date = split_date['Date'].apply(lambda a : str.split(a, "-"))
        split_date_df = pd.DataFrame(list(split_date))
        split_date_df = split_date_df.rename({0: "Year", 1: "Month", 2: "Day"}, axis=1)
        split_date_df['Day'] = split_date_df['Day'].astype(int)
        split_date_df['Month'] = split_date_df['Month'].astype(int)
        split_date_df['Year'] = split_date_df['Year'].astype(int)
        return split_date_df["Day"].values, split_date_df["Month"].values, split_date_df["Year"].values
    dataNoNA["Day"], dataNoNA["Month"], dataNoNA["Year"] = split_date(dataNoNA)
    
    # Adding season column
    dataNoNA["Summer"] = dataNoNA["Month"].apply(lambda a : 1 if a in [12, 1, 2] else 0)
    dataNoNA["Winter"] = dataNoNA["Month"].apply(lambda a : 1 if a in [6, 7, 8] else 0)
    dataNoNA["Spring"] = dataNoNA["Month"].apply(lambda a : 1 if a in [9, 10, 11] else 0)
    dataNoNA["Fall"] = dataNoNA["Month"].apply(lambda a : 1 if a in [3, 4, 5] else 0)

    # Train-test split (80-20)
    x_train, x_test, y_train, y_test = train_test_split(dataNoNA, labels, test_size=0.2, random_state=479)

    return [x_train, x_test, y_train, y_test]


preprocess_data()
