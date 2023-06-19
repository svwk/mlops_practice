#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# %% Импорт
import argparse
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

import message_constants as mc
from data_methods import preprocess_data, print_error

# %% Задание пути для сохранения файлов
script_path = os.getcwd()
models_path = None
data_path = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scriptsdir')
    parser.add_argument('-m', '--modelsdir')
    parser.add_argument('-d', '--datadir')
    namespace = parser.parse_args()
    if namespace.scriptsdir:
        script_path = namespace.scriptsdir
    if namespace.modelsdir:
        models_path = namespace.modelsdir
    if namespace.datadir:
        data_path = namespace.datadir

script_path = script_path.removesuffix("/")
models_path = script_path.replace('scripts', 'models') if models_path is None else models_path.removesuffix("/")
data_path = script_path.replace('scripts', 'data') if data_path is None else data_path.removesuffix("/")

# %% Если пути не существует, скрипт прерывает свою работу
if (not os.path.isdir(script_path)) or \
        (not os.path.isdir(data_path)) or \
        (not os.path.isdir(os.path.join(data_path, 'raw'))) or \
        (not os.path.isdir(os.path.join(data_path, 'processed'))) or \
        (not os.path.isdir(models_path)):
    print(mc.NO_PATH)
    sys.exit(4)

for filename in os.listdir(os.path.join(data_path, 'raw')):
    if filename.endswith(".csv"):
        full_filename = os.path.join(data_path, "raw", filename)
        scaler_filename = os.path.join(models_path, filename.replace('.csv', '_scaler.pkl'))
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        full_filename_processed = full_filename.replace("raw", "processed")

        try:
            dataset_data = pd.read_csv(full_filename, sep=',')
            dataset_data_stand = preprocess_data(dataset_data, scaler, True)
            dataset_data_stand.to_csv(full_filename_processed, index=False)
            print(f"{mc.STAND_SAVED}: {full_filename}")
            os.rename(full_filename, full_filename.replace("raw", "archive"))
            joblib.dump(scaler, scaler_filename)

        except Exception as ex:
            print_error(ex, full_filename)
            sys.exit( 5)
