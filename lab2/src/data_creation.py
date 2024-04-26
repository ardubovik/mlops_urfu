import numpy as np
import pandas as pd

from common import DataStorage as ds


# generation data (wind, temperature day, temperature night, precipitation)
def generate_data(days = 365):
    # anomaly days
    anomaly_days = np.random.choice(range(days), size=int(days * 0.5), replace=False)
    anomaly = np.zeros(days, dtype='int64')
    anomaly[anomaly_days] = 1
    
    # temperature
    daily_variation = np.random.normal(0, 2, size=days)
    daily_variation[anomaly_days] += np.random.normal(20, 5, size=len(anomaly_days))

    base_temperature_day = 15
    base_temperature_night = 1

    temp_day = base_temperature_day + daily_variation + np.sin(
        np.linspace(0, 2 * np.pi, days)
    ) * 10
    temp_night = base_temperature_night + daily_variation + np.sin(
        np.linspace(0, 2 * np.pi, days)
    ) * 10
    
    # precipitation 
    daily_precipitation = np.random.normal(0, 2, size=days)
    daily_precipitation[anomaly_days] += np.random.normal(30, 5, size=len(anomaly_days))
    precipitation = abs(daily_precipitation + np.sin(np.linspace(0, 2 * np.pi, days)))
    
    # wind
    wind = np.random.normal(2, 5, size=days)
    wind[anomaly_days] += np.random.normal(30, 2, size=len(anomaly_days))
    wind = abs(wind + np.sin(np.linspace(15, 2 * np.pi, days)))

    return temp_day, temp_night, precipitation, wind, anomaly


def main():
    np.random.seed(42)
    days = 365

    # generation train and test data
    train_temp_day, train_temp_night, train_precipitation, train_wind, anomaly = generate_data(
        days
    )
    test_temp_day, test_temp_night, test_precipitation, test_wind, anomaly = generate_data(days)

    # save data
    ds.save_data_to_csv(
        '../data/train',
        'train_data',
        pd.DataFrame({
            'temp_day': train_temp_day,
            'temp_night': train_temp_night,
            'precipitation': train_precipitation,
            'wind': train_wind,
            'anomaly_day': anomaly,
        })
    )
    
    ds.save_data_to_csv(
        '../data/test',
        'test_data',
        pd.DataFrame({
            'temp_day': test_temp_day,
            'temp_night': test_temp_night,
            'precipitation': test_precipitation,
            'wind': test_wind,
            'anomaly_day': anomaly,
        })
    )

if __name__ == '__main__':
    main()
