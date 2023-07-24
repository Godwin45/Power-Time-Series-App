import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

plt.style.use('fivethirtyeight')

class PredictionPipeline:
    def __init__(self, data_file, model_file):
        self.data_file = data_file
        self.model_file = model_file
        self.df = pd.read_csv(self.data_file)
        self.df = self.df.set_index('Datetime')
        self.df.index = pd.to_datetime(self.df.index)
        self.df = self.create_features(self.df)
        self.df = self.add_lags(self.df)
        self.future_w_features = None

    def create_features(self, df):
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

    def predict_future(self, start_date, end_date):
        future = pd.date_range(start_date, end_date, freq='1h')
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        self.df['isFuture'] = False
        df_and_future = pd.concat([self.df, future_df])
        df_and_future = self.create_features(df_and_future)
        df_and_future = self.add_lags(df_and_future)

        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                    'lag1', 'lag2', 'lag3']
        reg_new = xgb.XGBRegressor()
        reg_new.load_model(self.model_file)
        df_and_future['pred'] = reg_new.predict(df_and_future[FEATURES])
        self.future_w_features = df_and_future.query('isFuture').copy()
        return self.future_w_features
