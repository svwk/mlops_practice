#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import argparse
import os
import joblib
import sys

import message_constants as mc
from data_methods import test_model

# %% Задание пути для сохранения файлов
script_path = os.getcwd()
models_path = None
data_path = None
dataset_name = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scriptsdir')
    parser.add_argument('-m', '--modelsdir')
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-n', '--name')
    namespace = parser.parse_args()
    if namespace.scriptsdir:
        script_path = namespace.scriptsdir
    if namespace.modelsdir:
        models_path = namespace.modelsdir
    if namespace.datadir:
        data_path = namespace.datadir
    if namespace.name:
        dataset_name = namespace.name

script_path = script_path.removesuffix("/")
models_path = script_path.replace('scripts', 'models') if models_path is None else models_path.removesuffix("/")
data_path = script_path.replace('scripts', 'data') if data_path is None else data_path.removesuffix("/")

# %% Если пути не существует, скрипт прерывает свою работу
if (not os.path.isdir(script_path)) or \
        (not os.path.isdir(data_path)) or \
        (not os.path.isdir(os.path.join(data_path, 'test'))) or \
        (not os.path.isdir(models_path)):
    print(mc.NO_PATH)
    sys.exit(15)

files_test = [name for name in os.listdir(os.path.join(data_path, 'test'))
              if os.path.isfile(os.path.join(data_path, 'test', name))]
files_test_count = len(files_test)

if files_test_count == 0:
    print(mc.DIRECTORY_IS_EMPTY)
    sys.exit(16)

if dataset_name is None and files_test_count != 1:
    print(mc.NO_NAME)
    sys.exit(17)

filename = f"{dataset_name}.csv" if dataset_name is not None else None

if files_test_count == 1:
    if filename is None:
        filename = files_test[0]
    elif filename != files_test[0]:
        sys.exit(18)

full_filename_test = os.path.join(data_path, 'test', filename)
full_filename_model = os.path.join(models_path, filename.replace(".csv", "_model.pkl"))

try:
    dataset_test = pd.read_csv(full_filename_test, sep=',')
    model = joblib.load(full_filename_model)
    if model is None:
        print(mc.MODEL_READ_ERROR)
        sys.exit(19)

    accuracy = test_model(dataset_test, model)
except Exception as inst:
    print(f"{mc.MODEL_READ_ERROR} {full_filename_model} {inst.args}")
    sys.exit(20)
