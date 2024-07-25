import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.conponents.data_ingestion import DataIngestion

if __name__=="__main__":
    try:
        di = DataIngestion()
        di.initiate_data_ingestion()
    except Exception as e:
        raise CustomException(e, sys)