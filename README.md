# Augmenting Teleconferencing with Computer Vision
CMU INI Practicum

# Faculty
Piotr Mardziel, Carlee Joe-Wong

# Team members
Arpit Gupta, MengMeng Zhang, Nan Lin


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
sudo pip install pyqt5
sudo pip install v4l2
sudo pip install inotify
```

v4l2: Support for virtual cam on Linux
inotify: Support for file access event hook for autostart

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


# Undistorter Setup
You may want to undistort the video output if your camera has a wide angle lens.

To calibrate your camera, follow these steps:
1. Print out a picture of a chessboard.
2. Take several pictures using your camera with different orientations and save them in a folder.
Check pictures under [this folder](undistort/data/chessboard/original4) for reference.
3. Run the following commant to set up profile for you camera:
```
python3 undistortion.py <profile_name> <chessboard_path> <img_point_path> <obj_point_path> <device1> [<device2> ... <device n>]
```
* **profile_name**: The name of the profile, e.g.slight_640_800
* **chessboard_path**: Path to the folder of chessboard pictures, e.g."undistort/data/chessboard/original4/*"
* **img_point_path**: Path to save the img points without post fix, e.g."undistort/profiles/img1"
* **obj_point_path**: Path to save the obj points without post fix, e.g."undistort/profiles/obj1"
* **deviceN**: Device in the format of \<idVendor\>:\<idProduct\>, run lsusb to find them, e.g.05a3:9230
* e.g.python3 undistortion.py "undistort/data/chessboard/original4/*" "undistort/profiles/img1"
 "undistort/profiles/obj1" 05a3:9230


## Run
For the autostart watcher:
```
python3 event.py -h
```


or

For the main program:
```
python3 main.py -h
```
