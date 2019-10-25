g++ sr_video.cpp -o sr_video `pkg-config --cflags --libs opencv`
g++ sr_webcam.cpp -o sr_webcam `pkg-config --cflags --libs opencv`
g++ sr_frames.cpp -o sr_frames `pkg-config --cflags --libs opencv`
./sr_frames
