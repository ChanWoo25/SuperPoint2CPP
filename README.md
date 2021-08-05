SuperPoint feature extractor with cpp/c++ Implementation
======================================

> Hello, this is my old project.
> But, for some people that have a interest in applying superpoint featuer extractor to own project, I've written this README file.
> If this project was of little help to your project, please strike the star button. ***Enjoy!!***



## Prerequisite

In this project, following packages are used. Make sure that the right version of libraries is installed and linked.

1. Nvidia-driver & Cuda Toolkit 10.2 with cuDNN 7.6.5
> About Cuda toolkit installation, there are so many guides. Follow it! \
> **First** : sudo apt-get install nvidia-driver-440 (It is normal for a higher version of the secondary nvidia library to be installed. Just check that 'nvidia-driver-440' is installed.) \
> **Second** : Follow instructions in this link (https://developer.nvidia.com/cuda-10.2-download-archive)

2. OpenCV (C++) 3.4.11 version
``` shell
# Installation script
chmod +x INSTALL_OpenCV.sh
./INSTALL_OpenCV.sh
```

3. LibTorch 1.6.0 version (with GPU | Cuda Toolkit 10.2, cuDNN 7.6.5)
> If only CPU can be used, install cpu-version LibTorch. Some code change about tensor device should be required.
```shell
# Installation script
chmod +x INSTALL_LibTorch.sh
./INSTALL_LibTorch.sh
```

## BUILD

```shell
cmake -B build -S .
cmake --build build -t SuperPoint
```

## RUN
- You can use your webcam for fancy test. It'll be automatically detected in normal case.
```shell
# argument is frequncy
./bin/SuperPoint 100
```

### BibTeX Citation
```
@inproceedings{detone18superpoint,
  author    = {Daniel DeTone and
               Tomasz Malisiewicz and
               Andrew Rabinovich},
  title     = {SuperPoint: Self-Supervised Interest Point Detection and Description},
  booktitle = {CVPR Deep Learning for Visual SLAM Workshop},
  year      = {2018},
  url       = {http://arxiv.org/abs/1712.07629}
}
```
