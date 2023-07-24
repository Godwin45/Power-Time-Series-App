from powerTimeSeries.config.configuration import ConfigurationManager
from powerTimeSeries.components.training import Training
from powerTimeSeries import logger



STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
       
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.training()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
