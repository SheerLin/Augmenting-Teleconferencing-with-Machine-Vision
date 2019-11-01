# Augmenting Tele-conferencing with Computer Vision
CMU INI Practicum Team 5

# Faculaty
Piotr, Carlee

# Team members
Arpit Gupta, Nan Lin, MengMeng Zhang


# Build
## Virtual Driver
https://github.com/webcamoid/akvcam
```
git clone https://github.com/webcamoid/akvcam.git
cd akvcam/src
make
sudo make install
sudo depmod -a
```

### configuration
sudo vim /etc/akvcam/config.ini
```
[Cameras]
cameras/size = 2

cameras/1/type = output
cameras/1/mode = mmap, userptr, rw
cameras/1/description = Virtual Camera (output device)
cameras/1/formats = 2

cameras/2/type = capture
cameras/2/mode = mmap, rw
cameras/2/description = Virtual Camera
cameras/2/formats = 1, 2, 3

[Formats]
formats/size = 2

formats/1/format = YUY2, RGB32, BGR32
formats/1/width = 640
formats/1/height = 480
formats/1/fps = 30

formats/2/format = RGB24, BGR24
formats/2/width = 640
formats/2/height = 480
formats/2/fps = 30

formats/3/format = RGB24, BGR24
formats/3/width = 1920
formats/3/height = 1080
formats/3/fps = 60

[Connections]
connections/size = 1
connections/1/connection = 1:2
```

### insert mod
```
cd akvcam/src
sudo modprobe videodev
sudo insmod akvcam.ko
```

### remove mod
```
cd akvcam/src
    sudo rmmod akvcam.ko
```

## Python Lib
```
sudo pip install numpy
sudo pip install python-opencv
sudo pip install v4l2
sudo pip install inotify
```

Patch v4l2 for Python 3:
https://bugs.launchpad.net/python-v4l2/+bug/1664158
```
vim sudo vim /usr/local/lib/python3.7/site-packages/v4l2.py
Rather than
) = range(1, 9) + [0x80] 
the `v4l2_buf_type` line ought to be
) = list(range(1, 9)) + [0x80]

Rather than
) = range(0, 4) + [2]
the `v4l2_priority` line ought to be
) = list(range(0, 4)) + [2]
```

## Run
```
python main.py
```


## Docs
https://circuitdigest.com/tutorial/image-manipulation-in-python-opencv-part2
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html