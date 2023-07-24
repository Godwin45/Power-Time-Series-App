import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import xgboost as xgb
from powerTimeSeries.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config:TrainingConfig):
        self.config = config

    def training(self):
        df = pd.read_csv(self.config.training_data)

        self.train_model(df)

    def train_model(self, df):
        FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year',
                    'lag1', 'lag2', 'lag3']
        TARGET = 'PJME_MW'

        X_all = df[FEATURES]
        y_all = df[TARGET]

        print("Training process ongoing")

        reg = XGBRegressor(base_score=0.5,
                           booster='gbtree',
                           n_estimators=500,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
        reg.fit(X_all, y_all,
                eval_set=[(X_all, y_all)],
                verbose=100)
        

        self.save_model(reg)

    def save_model(self, model):
        model.save_model(self.config.trained_model_path)
        print("Model saved successfully.")
