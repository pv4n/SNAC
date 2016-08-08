#!/bin/bash
ffmpeg -framerate 30 -i ../frames/1_raw/25/frame%05d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p 1_raw/25.mp4
ffmpeg -framerate 30 -i ../frames/1_raw/30/frame%05d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p 1_raw/30.mp4
ffmpeg -framerate 30 -i ../frames/1_raw/35/frame%05d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p 1_raw/35.mp4
ffmpeg -framerate 30 -i ../frames/1_raw/40/frame%05d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p 1_raw/40.mp4
ffmpeg -framerate 30 -i ../frames/1_raw/45/frame%05d.jpeg -c:v libx264 -r 30 -pix_fmt yuv420p 1_raw/45.mp4

ffmpeg -framerate 30 -i ../frames/2_yolo/25/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 2_yolo/25.mp4
ffmpeg -framerate 30 -i ../frames/2_yolo/30/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 2_yolo/30.mp4
ffmpeg -framerate 30 -i ../frames/2_yolo/35/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 2_yolo/35.mp4
ffmpeg -framerate 30 -i ../frames/2_yolo/40/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 2_yolo/40.mp4
ffmpeg -framerate 30 -i ../frames/2_yolo/45/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 2_yolo/45.mp4

ffmpeg -framerate 30 -i ../frames/3_processed/25/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 3_processed/25.mp4
ffmpeg -framerate 30 -i ../frames/3_processed/30/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 3_processed/30.mp4
ffmpeg -framerate 30 -i ../frames/3_processed/35/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 3_processed/35.mp4
ffmpeg -framerate 30 -i ../frames/3_processed/40/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 3_processed/40.mp4
ffmpeg -framerate 30 -i ../frames/3_processed/45/frame%05d.jpeg_p.png -c:v libx264 -r 30 -pix_fmt yuv420p 3_processed/45.mp4