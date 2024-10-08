#!/bin/bash

# echo ""
# echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

# echo ""
# echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

# echo ""
# echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

# echo ""
# echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

# echo ""
# echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

# echo ""
# echo ""
echo "****************** Installing visdom ******************"
pip install visdom

# echo ""
# echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image

# echo ""
# echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

# echo ""
# echo ""
echo "****************** Installing gdown ******************"
pip install gdown

# echo ""
# echo ""
echo "****************** Installing cython ******************"
conda install -y cython

# echo ""
# echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

# echo ""
# echo ""
echo "****************** Installing LVIS toolkit ******************"
pip install lvis


# echo ""
# echo ""
echo "******** Installing spatial-correlation-sampler. Note: This is required only for KYS tracker **********"
pip install spatial-correlation-sampler

# echo ""
# echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

# echo ""
# echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

# echo ""
# echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

# echo ""
# echo ""
echo "****************** DiMP50 Network ******************"
gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth

