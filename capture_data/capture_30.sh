#!/bin/bash

#rm -rfv /home/ubuntu/Desktop/winlab/test.avi /home/ubuntu/Desktop/winlab/frames/

DATE_TIME="$(date '+%d-%b-%Y-%H-%M-%S')"
#ROOT_DIR=/home/ubuntu/Desktop/winlab/$DATE_TIME
ROOT_DIR=frames/$DATE_TIME
FRAME_RATE=30
JPEG_QUALITY=78

mkdir $ROOT_DIR

#streamer -q -c /dev/video0 -s 1280x720 -f rgb24 -r 30 -t 150 -j 75 -o test.avi
streamer -d -c /dev/video1 -s 1280x720 -r $FRAME_RATE -t 10:00:00 -j $JPEG_QUALITY -o $ROOT_DIR/frame00000.jpeg