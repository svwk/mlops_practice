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

if [[ ! -d "$install_dir" && ! -L "$install_dir" ]] ; then
  mkdir -m ug+rw -p "$install_dir"
  python3 -m venv "$install_dir"
fi

source "$install_dir/bin/activate"
pip install -r ./lab2/requirements.txt &> /dev/null
deactivate

 echo "Virtual Environment is installed"
