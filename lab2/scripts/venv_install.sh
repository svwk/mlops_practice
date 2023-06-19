#!/bin/bash

# Каталог для скриптов python
install_dir=$1

if [[ ! -d "$install_dir" && ! -L "$install_dir" ]] ; then
    echo "The specified install directory does not exist."
    echo " The program is over."
    exit 4
fi

if [ !  -r "$install_dir" ] ; then
  echo "The  specified install  directory cannot be accessed."
  echo "The program is over."
  exit 5
fi

# Если venv не установлен:
if [ $(dpkg-query -W -f='${Status}' 'python3-venv' | grep -c 'ok installed') -eq 0 ];
then
  apt-get install python3-venv
fi

if [ $(dpkg-query -W -f='${Status}' 'python3-pip' | grep -c 'ok installed') -eq 0 ];
then
 apt-get install -y python3-pip
fi

venv_dir="$install_dir/env"

if [[ ! -d "$venv_dir" && ! -L "$venv_dir" ]] ; then
  python3 -m venv "$venv_dir"
fi

source "$venv_dir/bin/activate"
pip install -r requirements.txt &> /dev/null
deactivate

 echo "Virtual Environment is installed"
