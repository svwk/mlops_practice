#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# %% Импортируем нужные библиотеки
import argparse
import os
import sys
import pandas as pd
import joblib

import message_constants as mc
from data_methods import train_model, print_error

# %% Задаем путь для сохранения модели, а так же пути для чтения датасета
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
        (not os.path.isdir(os.path.join(data_path, 'train'))) or \
        (not os.path.isdir(models_path)):
    print(mc.NO_PATH)
    sys.exit(10)

files_train = [name for name in os.listdir(os.path.join(data_path, 'train'))
         if os.path.isfile(os.path.join(data_path, 'train', name))]
files_train_count = len(files_train)

if files_train_count == 0:
    print(mc.DIRECTORY_IS_EMPTY)
    sys.exit(11)

if dataset_name is None and files_train_count != 1:
    print(mc.NO_NAME)
    sys.exit(12)

filename = f"{dataset_name}.csv" if dataset_name is not None else None

if files_train_count == 1:
    if filename is None:
        filename = files_train[0]
    elif filename != files_train[0]:
        sys.exit(13)

full_filename_train = os.path.join(data_path, 'train', filename)
full_filename_model = os.path.join(models_path, filename.replace(".csv", "_model.pkl"))

try:
    dataset_train  = pd.read_csv(full_filename_train, sep=',')
    model = train_model(dataset_train)
    joblib.dump(model, full_filename_model)
    print(f"{mc.MODEL_SAVED}: {full_filename_model}")
except Exception as ex:
    print_error(ex, filename)
    sys.exit(14)
