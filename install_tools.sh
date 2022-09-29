sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y update
sudo apt-get -y install python3.9
sudo apt-get -y install python3.9-dev
sudo apt-get -y install python3-pip
sudo apt-get -y install python3.9-distutils
python3.9 -m pip install --upgrade setuptools
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --upgrade distlib
python3.9 -m pip install scipy numpy pandas sentencepiece torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116 transformers fugashi[unidic-lite]