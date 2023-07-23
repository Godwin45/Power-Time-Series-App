import os
from powerTimeSeries.entity.config_entity import DataTransformationConfig
import pandas as pd

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        self.config = config
       
       
    def transformation_data(self):

        df = pd.read_csv(self.config.data)
        df = df.set_index('Datetime')
        df.index = pd.to_datetime(df.index)
        

        df = self.create_features(df)
        df = self.add_lags(df)

        self.save(df)


    def create_features(self, df):
        """
        Create time series features based on time series index.
        """
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week

        return df
    
    def add_lags(self, df):
        target_map = df['PJME_MW'].to_dict()
        df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
        df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
        df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)

        return df
    def save(self, df):
        # Create the 'transformed_data' directory
        transformed_data_dir = os.path.join("artifacts", "transformed_data")
        os.makedirs(transformed_data_dir, exist_ok=True)

        # Save the DataFrame to the 'data.csv' file inside the 'transformed_data' directory
        data_file_path = os.path.join(transformed_data_dir, "data.csv")
        df.to_csv(data_file_path, index=False)

        print("Data transformation complete")
