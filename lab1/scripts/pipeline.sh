#!/bin/bash

# сhecking the activation virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "virtual environment is not activate"

    # create virtual environment
    echo "create virtual environment"
    python3 -m venv ../../venv || { echo "error when creating a virtual environment"; exit 1; }

    # activating virtual environment
    echo "activating virtual environment"
    source ../../venv/bin/activate || { echo "error when activating a virtual environment"; exit 1; }

    echo "install requirements"
    pip install -r ../requirements.txt || { echo "error when installing requirements"; exit 1; }
else
    echo "virtual environment is activated"
fi

cd ../src

echo "starting data creation"
python data_creation.py || { echo "error during execution data_creation.py"; exit 1; }

echo "starting model preprocessing"
python model_preprocessing.py || { echo "error during execution model_preprocessing.py"; exit 1; }

echo "starting model preparation"
python model_preparation.py || { echo "error during execution model_preparation.py"; exit 1; }

echo "starting model testing"
python model_testing.py || { echo "error during execution model_testing.py"; exit 1; }