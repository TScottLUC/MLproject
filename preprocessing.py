import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# So that we may see all columns
pd.set_option('display.max_columns', None)


def preprocess_data():
    # Read in the data
    data = pd.read_csv("weatherAUS.csv")

    # Convert 9pm/3pm data values to "delta" values and drop the originals from the dataframe
    data["deltaWindSpeed"] = data["WindSpeed3pm"] - data["WindSpeed9am"]
    data["deltaHumidity"] = data["Humidity3pm"] - data["Humidity9am"]
    data["deltaPressure"] = data["Pressure3pm"] - data["Pressure9am"]
    data["deltaCloud"] = data["Cloud3pm"] - data["Cloud9am"]
    data["deltaTemp"] = data["Temp3pm"] - data["Temp9am"]
    data["RainToday"] = np.where(data["RainToday"] == "Yes", 1, 0)
    data["RainTomorrow"] = np.where(data["RainTomorrow"] == "Yes", 1, 0)
    data = data.drop(
        ["Location", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
         "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm"], axis=1)

    data = data.interpolate()

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
    dataNoNA = (dataNoNA - dataNoNA.mean()) / dataNoNA.std()

    dataNoNA["Date"] = dateData

    # Train-test split (80-20)
    x_train, x_test, y_train, y_test = train_test_split(dataNoNA, labels, test_size=0.2, random_state=479)

    return [x_train, x_test, y_train, y_test]


preprocess_data()
