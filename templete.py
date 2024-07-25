import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')




list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/conponents/__init__.py",
    f"src/conponents/data_ingestion.py",
    f"src/conponents/data_cleaning.py",
    f"src/conponents/data_tokenization.py",
    f"src/conponents/model_building.py",
    f"src/conponents/model_training.py",
    f"src/conponents/model_evaluation.py",
    f"src/logger/__init__.py",
    f"src/exception/__init__.py",
    f"app/__init__.py",
    f"app/routes.py",
    f"app/pipeline.py",
    f"templets/index.html",
    "models",
    "data",
    "requirements.txt",
    "setup.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py"


]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    
    else:
        logging.info(f"{filename} is already exists")