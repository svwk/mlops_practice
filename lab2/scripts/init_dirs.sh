#!/bin/bash

# Каталог для скриптов python
script_dir=$1

# Каталог для данных
data_dir=$2

# Каталог для модели
models_dir=$3

if [[ ! -d "$script_dir" && ! -L "$script_dir" ]] ; then
    echo "The specified script directory does not exist."
    echo " The program is over."
    exit 1
fi

if [ !  -r "$script_dir" ] ; then
  echo "The scripts directory cannot be accessed."
  echo "The program is over."
  exit 2
fi

mkdir -m ug+rw -p "$models_dir"
echo "$models_dir   created"
mkdir -m ug+rw -p "$data_dir"
mkdir -m ug+rw -p "$data_dir/archive"
mkdir -m ug+rw -p "$data_dir/processed"
mkdir -m ug+rw -p "$data_dir/raw"
mkdir -m ug+rw -p "$data_dir/test"
mkdir -m ug+rw -p "$data_dir/train"
echo "$data_dir   created"

if [[  ! -r "$data_dir" || ! -w "$data_dir" ]] ; then
  echo "The data directory cannot be accessed."
  echo "The program is over."
  exit 3
fi
