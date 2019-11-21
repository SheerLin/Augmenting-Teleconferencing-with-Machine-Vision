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
cameras/1/formats = 2, 4

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

formats/4/format = RGB24, BGR24
formats/4/width = 1920
formats/4/height = 1080
formats/4/fps = 30

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
sudo pip install pyqt5
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

## Before Running
You may want to undistort the frames if you use a USB camera.

To calibrate your camera, you will need to:
1. Print out a chessboard picture.
2. Take several pictures using your USB camera and put them in a folder that is accessible. 
Check pictures under 
[this folder](undistort/data/chessboard/original4) as reference.
3. Run this program to set up profile for you camera:
```
python3 undistortion.py <chessboard path> <img point path> <obj point path> <device1> [<device2> ... <device n>]
```
* **chessboard path**: Path to the folder of chessboard pictures, e.g."undistort/data/chessboard/original4/*"
* **img point path**: Path to save the img points without post fix, e.g."undistort/profiles/img1"
* **obj point path**: Path to save the obj points without post fix, e.g."undistort/profiles/obj1"
* **device n**: Device in the format of \<idVendor\>:\<idProduct\>, run lsusb to find them, e.g.05a3:9230
* e.g.python3 undistortion.py "undistort/data/chessboard/original4/*" "undistort/profiles/img1"
 "undistort/profiles/obj1" 05a3:9230



## Run
```
python3 main.py
```


## Docs
https://circuitdigest.com/tutorial/image-manipulation-in-python-opencv-part2
https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html