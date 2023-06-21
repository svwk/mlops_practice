#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# %% Импорт
import argparse
import os
import sys
import pandas as pd

import message_constants as mc
from data_methods import separate_dataset, print_error

# %% Задание пути для сохранения файлов
script_path = os.getcwd()
models_path = None
data_path = None
dataset_name = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scriptsdir')
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-n', '--name')
    namespace = parser.parse_args()
    if namespace.scriptsdir:
        script_path = namespace.scriptsdir
    if namespace.datadir:
        data_path = namespace.datadir
    if namespace.name:
        dataset_name = namespace.name

script_path = script_path.removesuffix("/")
data_path = script_path.replace('scripts', 'data') if data_path is None else data_path.removesuffix("/")

# %% Если пути не существует, скрипт прерывает свою работу
if (not os.path.isdir(script_path)) or \
        (not os.path.isdir(data_path)) or \
        (not os.path.isdir(os.path.join(data_path, 'train'))) or \
        (not os.path.isdir(os.path.join(data_path, 'test'))) or \
        (not os.path.isdir(os.path.join(data_path, 'processed'))):
    print(mc.NO_PATH)
    sys.exit(6)

files = [name for name in os.listdir(os.path.join(data_path, 'processed'))
         if os.path.isfile(os.path.join(data_path, 'processed', name))]
files_count = len(files)

if files_count == 0:
    print(mc.DIRECTORY_IS_EMPTY)
    sys.exit(7)

if dataset_name is None and files_count != 1:
    print(mc.NO_NAME)
    sys.exit(8)

filename = f"{dataset_name}.csv" if dataset_name is not None else None

if files_count == 1:
    if filename is None:
        filename = files[0]
    elif filename != files[0]:
        sys.exit(5)

full_filename_processed = os.path.join(data_path, 'processed', filename)
full_filename_train = os.path.join(data_path, 'train', filename)
full_filename_test = os.path.join(data_path, 'test', filename)
try:
    dataset_data = pd.read_csv(full_filename_processed, sep=',')
    dataset_train, dataset_test = separate_dataset(dataset_data)

    dataset_train.to_csv(full_filename_train, index=False)
    dataset_test.to_csv(full_filename_test, index=False)

    print(f"{mc.SOURCE_SAVED}: {filename}")
except Exception as ex:
    print_error(ex, filename)
    sys.exit(9)
