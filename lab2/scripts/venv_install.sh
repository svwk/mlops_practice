#!/bin/bash

# Каталог для скриптов python
install_dir=$1

# Если venv не установлен:
if [ $(dpkg-query -W -f='${Status}' 'python3-venv' | grep -c 'ok installed') -eq 0 ];
then
  apt-get install python3-venv
fi

if [ $(dpkg-query -W -f='${Status}' 'python3-pip' | grep -c 'ok installed') -eq 0 ];
then
 apt-get install -y python3-pip
fi

if [[ ! -d "$venv_dir" && ! -L "$venv_dir" ]] ; then
  mkdir -m ug+rw -p "$venv_dir"
  python3 -m venv "$venv_dir"
fi

source "$venv_dir/bin/activate"
pip install -r requirements.txt &> /dev/null
deactivate

 echo "Virtual Environment is installed"
