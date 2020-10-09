#! /bin/bash

cd ~

echo "Download eigen 3.3.8 version zip file..."
wget -O eigen3.zip https://gitlab.com/libeigen/eigen/-/archive/3.3.8/eigen-3.3.8.zip

# INSTALL EIGEN3
echo "Install Eigen3..."
unzip eigen3.zip
cd eigen-3.3.8
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
sudo make install && cd ../..


echo "Download 3.4.11 version OpenCV and OpenCV Contrib..."
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.11.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.11.zip

# INSTALL OpenCV
echo "Install OpenCV with some options..."
sudo unzip opencv.zip -d /usr/local
sudo unzip opencv_contrib.zip -d /usr/local
sudo apt install git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

cd /usr/local/opencv-3.4.11
sudo mkdir build
cd build

sudo cmake -DBUILD_opencv_world -DCMAKE_CONFIGURATION_TYPE=Release -DOPENCV_ENABLE_NONFREE -DOPENCV_EXTRA_MODULES_PATH=/usr/local/opencv_contrib-3.4.11/modules -DCMAKE_INSTALL_PREFIX=/usr/local ..
sudo make -j$(nproc)
sudo make install

# Remove .zip files.
# cd ~ && rm eigen3.zip opencv.zip opencv_contrib.zip 
