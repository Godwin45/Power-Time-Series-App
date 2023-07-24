from powerTimeSeries.constants import *
import os
import pandas as pd
from powerTimeSeries.utils.common import read_yaml, create_directories
from powerTimeSeries.entity.config_entity import (DataIngestionConfig,
                                                  DataTransformationConfig,
                                                  TrainingConfig)
                                                


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data=Path(config.data),
          
        )

        return  data_transformation_config

    def get_training_config(self) -> TrainingConfig:
        config = self.config.training

        create_directories([
                Path(config.root_dir)
            ])

        training_config = TrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_path=Path(config.trained_model_path),
            training_data=Path(config.training_data),
        )

        return training_config
