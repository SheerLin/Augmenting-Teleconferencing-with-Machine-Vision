g++ sr_video.cpp -o sr_video `pkg-config --cflags --libs opencv`
g++ sr_webcam.cpp -o sr_webcam `pkg-config --cflags --libs opencv`
g++ sr_frames.cpp -o sr_frames `pkg-config --cflags --libs opencv`
./sr_frames

# test Digital Ocena 1vCPU 1G mem, seconds
Elapsed time: 57.502seconds, frame =0
Elapsed time: 73.7906seconds, frame =1
Elapsed time: 89.3775seconds, frame =2
Elapsed time: 105.185seconds, frame =3
Elapsed time: 121.437seconds, frame =4
Elapsed time: 137.036seconds, frame =5
Elapsed time: 152.916seconds, frame =6
Elapsed time: 168.087seconds, frame =7
Elapsed time: 183.917seconds, frame =8
Elapsed time: 199.275seconds, frame =9

# test on mac ubuntu vm, 4cpu, 8G mem, seconds
/* Elapsed time: 14.6527 count =1
Elapsed time: 12.3023seconds, frame =0
Elapsed time: 17.1125seconds, frame =1
Elapsed time: 22.0923seconds, frame =2
Elapsed time: 27.7308seconds, frame =3
Elapsed time: 33.3677seconds, frame =4
Elapsed time: 39.2368seconds, frame =5
Elapsed time: 45.6676seconds, frame =6
Elapsed time: 52.0725seconds, frame =7
Elapsed time: 60.3158seconds, frame =8
Elapsed time: 68.7447seconds, frame =9*/

# small pc
Elapsed time: 47.8281seconds, frame =0
Elapsed time: 58.7665seconds, frame =1
Elapsed time: 69.6879seconds, frame =2
Elapsed time: 80.4911seconds, frame =3
Elapsed time: 91.3095seconds, frame =4
Elapsed time: 102.171seconds, frame =5
Elapsed time: 113.035seconds, frame =6
Elapsed time: 123.899seconds, frame =7
Elapsed time: 134.722seconds, frame =8
Elapsed time: 145.519seconds, frame =9
Elapsed time: 156.358seconds, frame =10