#include <cxcore.h>
#include <cv.h>
#include <fstream>
#include <highgui.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

// #include <opencv2/imgcodecs.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"

extern "C"
{
	#include "src/imgfeatures.h"
	#include "src/kdtree.h"
	#include "src/sift.h"
	#include "src/utils.h"
	#include "src/xform.h"
}

// the maximum number of keypoint NN candidates to check during BBF search
#define KDTREE_BBF_MAX_NN_CHKS 200

// threshold on squared ratio of distances between NN and 2nd NN
#define NN_SQ_DIST_RATIO_THR 0.49

#define MINIMUM_FRAMES 3
#define MINIMUM_FEATURES 10
#define XMAX_THRES 640
#define YMAX_THRES 650

#define START_FRAME 2600
#define END_FRAME 4399

#define SPEED_FILE 40

#define DISTANCE 0.6 // miles
#define RECALCULATE_FRAMES 900
#define LANES 3

#define BBOX_FILE "../data/1_minute_light_traffic/bbox_files/car.txt"
#define RAW_FRAMES_LOCATION "../data/1_minute_light_traffic/frames/1_raw"
#define PROCESSED_FRAMES_LOCATION "../data/1_minute_light_traffic/frames/2_yolo"
#define COUNTED_FRAMES_LOCATION "../data/1_minute_light_traffic/frames/3_processed"

#define LOG_FILE_NAME "../data/1_minute_light_traffic/logs/1_minute_light_traffic.txt"

using namespace cv;
using namespace std;

double limit_value(double x, double a, double b)
{
	return ((x < a) ? a : ((x > b) ? b : x));
}
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color)
{
	for(int y = 0; y < cflowmap.rows; y += step)
	{
		for(int x = 0; x < cflowmap.cols; x += step)
		{
			const Point2f& fxy = flow.at<Point2f>(y, x);
			line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
			circle(cflowmap, Point(x,y), 2, color, -1);
		}
	}
}

void peekline(ifstream &is, string &s)
{
	streampos sp = is.tellg();
	getline(is, s);
	is.seekg(sp);
}

int main(int argc, char** argv)
{
	Mat image;

	long int image_index = START_FRAME;
	long int frame_number = START_FRAME;
	string traffic_class = "Traffic class: N/A";
	string current_car_speed = "Current Car Speed: N/A";
	string average_car_speed = "Average Car Speed: N/A";
	int prev_car_count = 0;

	float average_speed = 0;
	int average_speed_count = 0;

	// Read bounding box data
	ifstream bbox_file(BBOX_FILE);

	// Vector of vectors of vectors for bbox data
	vector<vector<vector<float>>> bboxes(END_FRAME - START_FRAME);

	float xmin, ymin, xmax, ymax;

	vector<double> flux(END_FRAME - START_FRAME);
	vector<double> length_arr(END_FRAME - START_FRAME);
	vector<double> bbox_heights(END_FRAME - START_FRAME);
	vector<double> speed(END_FRAME - START_FRAME);
	vector<double> center_x_arr(END_FRAME - START_FRAME);
	vector<double> center_y_arr(END_FRAME - START_FRAME);
	vector<double> depth_arr(END_FRAME - START_FRAME);


	// Read bbox data from file
	do
	{
		// Check if frame number in bbox file matches frame index
		string line;
		peekline(bbox_file, line);
		sscanf(line.c_str(), "frame%ld", &frame_number);

		xmin = limit_value(xmin, 0, 1280);
		ymin = limit_value(ymin, 0, 720);
		xmax = limit_value(xmax, 0, 1280);
		ymax = limit_value(ymax, 0, 720);

		// If yes, then continue
		if (frame_number == image_index)
		{
			getline(bbox_file, line);
			vector<float> current_bbox(4);
			sscanf(line.c_str(), "%*s %*f %f %f %f %f", &xmin, &ymin, &xmax, &ymax);
			current_bbox[0] = xmin;
			current_bbox[1] = ymin;
			current_bbox[2] = xmax;
			current_bbox[3] = ymax;

			bboxes[frame_number - START_FRAME].push_back(current_bbox);

			// printf("frame: %d\txmin: %f\tymin: %f\txmax: %f\tymax: %f\t\n", image_index, xmin, ymin, xmax, ymax);
		}
		else
		{
			image_index++;
		}
	} while (image_index < END_FRAME);

	bbox_file.close();

	// Sort all bbox vectors by leftmost x coordinate
	for (int f=START_FRAME; f<END_FRAME; f++)
	{
		std::sort(bboxes[f-START_FRAME].begin(), bboxes[f-START_FRAME].end(),
			[](const std::vector<float> &a, const std::vector<float> &b)
			{
				return a[0] < b[0];
			});
	}

	// Print out bbox data for confirmation
	/*for (int f=START_FRAME; f<END_FRAME; f++)
	{
		int i = 0;
		for (auto bbox : bboxes[f-START_FRAME])
		{
			i++;
			printf("Frame: %d\tbbox: %d\txmin: %f\tymin %f\txmax: %f\tymax %f\n", f, i, bbox[0], bbox[1], bbox[2], bbox[3]);
		}
	} */

	int matched_frames = 0;
	int car_count = 0;

	Mat flow, cflow, frame;
	Mat gray, prevgray, uflow;

	for (long int f=START_FRAME+1; f<END_FRAME; f++)
	{
		char image_file_zero[100];
		sprintf(image_file_zero, RAW_FRAMES_LOCATION"/frame0%04ld.jpeg", f-1);

		// printf("%s\n", image_file_zero);
		Mat image_zero = imread(image_file_zero, CV_LOAD_IMAGE_COLOR);


		if(bboxes[f-START_FRAME-1].empty())
		{
			// printf("SKIPPED\n");
			char image_file_processed[100];
			char image_file_processed_new[100];
			sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);

			if (((f - START_FRAME) % RECALCULATE_FRAMES) == 0)
			{
				double tc_speed = 140 - (car_count - prev_car_count) / DISTANCE / 2;
				string tc = "-1";

				if (tc_speed > 123.33)			{	tc = "A";	}
				else if (tc_speed > 106.66) 	{	tc = "B";	}
				else if (tc_speed > 90) 		{	tc = "C";	}
				else if (tc_speed > 73.33) 		{	tc = "D";	}
				else if (tc_speed > 56.66) 		{	tc = "E";	}
				else if (tc_speed > 40) 		{	tc = "F";	}
				else if (tc_speed > 23.33) 		{	tc = "7";	}
				else if (tc_speed > 6.66) 		{	tc = "8";	}
				else if (tc_speed > -10) 		{	tc = "9";	}
				else 							{	tc = "10";}

				traffic_class = "Traffic class: " + tc;
				prev_car_count = car_count;
			}

			// traffic_class = "Traffic class: N/A";
			current_car_speed = "Current Car Speed: N/A";
			string depth_string = "Current Car Depth: N/A";

			putText(image_processed, traffic_class, cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, depth_string, cvPoint(10, 80), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, current_car_speed, cvPoint(10, 110), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, average_car_speed, cvPoint(10, 140), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			imshow("traffic", image_processed);

			try
			{
				// printf("WRITING IMAGE\n");
				imwrite(image_file_processed_new, image_processed, compression_params);
			}
			catch (runtime_error& ex)
			{
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				return 1;
			}

			// Uncomment to wait for keypress when match is detected
			cvWaitKey(1);

			continue;
		}
		cv::Rect bound_box_zero(bboxes[f-START_FRAME-1][0][0],
								bboxes[f-START_FRAME-1][0][1],
								bboxes[f-START_FRAME-1][0][2] - bboxes[f-START_FRAME-1][0][0],
								bboxes[f-START_FRAME-1][0][3] - bboxes[f-START_FRAME-1][0][1]);
		cv::Mat cropped_zero = image_zero(bound_box_zero);
		IplImage img1_image = cropped_zero;
		IplImage* img1 = &img1_image;

		char image_file_one[100];
		sprintf(image_file_one, RAW_FRAMES_LOCATION"/frame0%04ld.jpeg", f);
		// printf("%s\n", image_file_one);
		Mat image_one = imread(image_file_one, CV_LOAD_IMAGE_COLOR);

		if(bboxes[f-START_FRAME].empty())
		{
			// printf("SKIPPED2\n");
			char image_file_processed[100];
			char image_file_processed_new[100];
			sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);

			if (((f - START_FRAME) % RECALCULATE_FRAMES) == 0)
			{
				double tc_speed = 140 - (car_count - prev_car_count) / DISTANCE / 2;
				string tc = "-1";

				if (tc_speed > 123.33)			{	tc = "A";	}
				else if (tc_speed > 106.66) 	{	tc = "B";	}
				else if (tc_speed > 90) 		{	tc = "C";	}
				else if (tc_speed > 73.33) 		{	tc = "D";	}
				else if (tc_speed > 56.66) 		{	tc = "E";	}
				else if (tc_speed > 40) 		{	tc = "F";	}
				else if (tc_speed > 23.33) 		{	tc = "7";	}
				else if (tc_speed > 6.66) 		{	tc = "8";	}
				else if (tc_speed > -10) 		{	tc = "9";	}
				else 							{	tc = "10";}

				traffic_class = "Traffic class: " + tc;
				prev_car_count = car_count;
			}

			// traffic_class = "Traffic class: N/A";
			current_car_speed = "Current Car Speed: N/A";
			string depth_string = "Current Car Depth: N/A";

			putText(image_processed, traffic_class, cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, depth_string, cvPoint(10, 80), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, current_car_speed, cvPoint(10, 110), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, average_car_speed, cvPoint(10, 140), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			imshow("traffic", image_processed);

			try
			{
				// printf("WRITING IMAGE\n");
				imwrite(image_file_processed_new, image_processed, compression_params);
			}
			catch (runtime_error& ex)
			{
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				return 1;
			}

			// Uncomment to wait for keypress when match is detected
			cvWaitKey(1);
			continue;
		}
		cv::Rect bound_box_one(bboxes[f-START_FRAME][0][0],
								bboxes[f-START_FRAME][0][1],
								bboxes[f-START_FRAME][0][2] - bboxes[f-START_FRAME][0][0],
								bboxes[f-START_FRAME][0][3] - bboxes[f-START_FRAME][0][1]);
		cv::Mat cropped_one = image_one(bound_box_one);
		IplImage img2_image = cropped_one;
		IplImage* img2 = &img2_image;


		IplImage *stacked;
		struct feature* feat1, * feat2, * feat;
		struct feature** nbrs;
		struct kd_node* kd_root;
		CvPoint pt1, pt2;
		double d0, d1;
		int n1, n2, k, i, m = 0;

		stacked = stack_imgs(img1, img2);

		// fprintf( stderr, "Finding features in %s...\n", "CROPPED ONE");
		n1 = sift_features(img1, &feat1);
		// fprintf( stderr, "Finding features in %s...\n", "CROPPED TWO" );
		n2 = sift_features(img2, &feat2);
		// fprintf( stderr, "Building kd tree...\n" );
		kd_root = kdtree_build(feat2, n2);
		for(i = 0; i < n1; i++)
			{
				feat = feat1 + i;
				k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
				if(k == 2)
				{
					d0 = descr_dist_sq(feat, nbrs[0]);
					d1 = descr_dist_sq(feat, nbrs[1]);
					if(d0 < d1 * NN_SQ_DIST_RATIO_THR)
						{
							pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
							pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
							pt2.y += img1->height;
							cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
							m++;
							feat1[i].fwd_match = nbrs[0];
						}
				}
				free(nbrs);
			}

		// fprintf(stderr, "Found %d total matches\n", m);

		// Uncomment to display bboxes and matches
		// char *matches = "Matches";
		// display_big_img(stacked, matches);
		// cvWaitKey(1);

		printf("%ld / %d\n", f, END_FRAME);

		if (m > MINIMUM_FEATURES)
		{
			matched_frames++;
		}
		else
		{
			matched_frames = 0;
		}

		if (matched_frames >= MINIMUM_FRAMES)
		{
			if (matched_frames == MINIMUM_FRAMES)
			{
				car_count++;
			}

			if ((bboxes[f-START_FRAME][0][2] < XMAX_THRES) && (bboxes[f-START_FRAME][0][3] < YMAX_THRES))
			{

				int center_x = (bboxes[f-START_FRAME-1][0][0] + bboxes[f-START_FRAME-1][0][2]) / 2;
				// center_x += 50;
				int center_y = (bboxes[f-START_FRAME-1][0][1] + bboxes[f-START_FRAME-1][0][3]) / 2;

				center_x_arr[f-START_FRAME] = center_x;
				center_y_arr[f-START_FRAME] = center_y;

				// printf("Found unique car #%d\n", car_count);

				char image_file_processed[100];
				char image_file_processed_new[100];
				sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
				sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
				Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);
				putText(image_processed, to_string(car_count), cvPoint(bboxes[f-START_FRAME][0][2], bboxes[f-START_FRAME][0][3]), FONT_HERSHEY_SIMPLEX, 2, cvScalar(0, 0, 255), 5, CV_AA);

				printf("Flow: %f\n", length_arr[f-START_FRAME-1]);
				printf("BBox Height: %f\n", bbox_heights[f-START_FRAME-1]);
				printf("center_x: %f\n", center_x_arr[f-START_FRAME-1]);
				printf("center_y: %f\n", center_y_arr[f-START_FRAME-1]);
				printf("depth: %f\n\n", depth_arr[f-START_FRAME-1]);

				// % run_nn {flow} {bbox height} {center x} {center y} {depth}
				char octave_call[200];
				sprintf(octave_call, "octave --path run_nn --silent run_nn/run_nn.m %f %f %f %f %f > speed",
					length_arr[f-START_FRAME-1], bbox_heights[f-START_FRAME-1], center_x_arr[f-START_FRAME-1], center_y_arr[f-START_FRAME-1], depth_arr[f-START_FRAME-1]);

				// char* octave_call = "octave --path run_nn --silent run_nn/run_nn.m 7 136 225 418 43 > speed";
				// sprintf(octave_call, "octave run_nn.m %f %f %f %f %f > class.txt", , , , )
				int a = system(octave_call);
				if (a == -1)
				{
					exit(0);
				}

				// traffic_class = "Traffic class: N/A";
				current_car_speed = "Current Car Speed: N/A";

				ifstream file("speed");
				string file_speed;
				getline(file, file_speed);
				// printf("speed: %s\n", file_speed.c_str());
				current_car_speed = "Current Car Speed: " + file_speed;

				printf("average_speed_count: %d\n", average_speed_count);

				average_speed_count++;
				average_speed = average_speed * (average_speed_count-1);
				average_speed += atoi(file_speed.c_str());
				average_speed = average_speed / average_speed_count;
				average_car_speed = "Average Car Speed: " + to_string(average_speed);

				// FILE *lsofFile_p = popen("octave run_nn.m 7 136 225 418 43", "r");

				// if (!lsofFile_p)
				// {
				//   return -1;
				// }

				// char buffer[1024];
				// char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p);
				// pclose(lsofFile_p);

				// printf("%s\n", line_p);

				if (((f - START_FRAME) % RECALCULATE_FRAMES) == 0)
				{
					double tc_speed = 140 - (car_count - prev_car_count) / DISTANCE / 2;
					string tc = "-1";

					if (tc_speed > 123.33)			{	tc = "A";	}
					else if (tc_speed > 106.66) 	{	tc = "B";	}
					else if (tc_speed > 90) 		{	tc = "C";	}
					else if (tc_speed > 73.33) 		{	tc = "D";	}
					else if (tc_speed > 56.66) 		{	tc = "E";	}
					else if (tc_speed > 40) 		{	tc = "F";	}
					else if (tc_speed > 23.33) 		{	tc = "7";	}
					else if (tc_speed > 6.66) 		{	tc = "8";	}
					else if (tc_speed > -10) 		{	tc = "9";	}
					else 							{	tc = "10";}

					traffic_class = "Traffic class: " + tc;
					prev_car_count = car_count;
				}

				putText(image_processed, traffic_class, cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
				putText(image_processed, current_car_speed, cvPoint(10, 110), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
				putText(image_processed, average_car_speed, cvPoint(10, 140), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);

				// try
				// {
				// 	imwrite(image_file_processed_new, image_processed, compression_params);
				// }
				// catch (runtime_error& ex)
				// {
				// 	fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				// 	return 1;
				// }

				char input[100];
				sprintf(input, RAW_FRAMES_LOCATION"/frame%05ld.jpeg", f);
				Mat current_image = imread(input, CV_LOAD_IMAGE_COLOR);
				cvtColor(current_image, gray, COLOR_BGR2GRAY);

				bbox_heights[f-START_FRAME] = bboxes[f-START_FRAME][0][3] - bboxes[f-START_FRAME][0][1];

				float depth = 5027.2211442592 * pow(bbox_heights[f - START_FRAME], -0.9659360221);
				depth_arr[f - START_FRAME] = depth;
				string depth_string = "Current Car Depth: " + to_string(depth);
				putText(image_processed, depth_string, cvPoint(10, 80), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

				if( !prevgray.empty() )
				{
					// calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
					calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 20, 3, 7, 1.5, 0);
					cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
					uflow.copyTo(flow);
					drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
					// imshow("flow", cflow);
					const Point2f& fxy = flow.at<Point2f>(center_y, center_x);
					float length = sqrt(pow(fxy.x, 2) + pow(fxy.y, 2));
					// printf("Flow at (%d, %d):\t%f\n\n", center_x, center_y, length);

					// double ratio = 0;
					double last_five_avg = 0;

					if (f-START_FRAME > 3)
					{
						double sum = 0;
						double bbox_box_sum = 0;
						for (int i=0; i < 3; i++)
						{
							sum += length_arr[f-START_FRAME - i];
							bbox_box_sum += bbox_heights[f-START_FRAME - i];
						}

						// printf("sum: %f\n", sum);
						//ratio = 35639.6340611629 * length / (bbox_box_sum * bbox_box_sum);
						last_five_avg = sum / 3;

						// printf("last_five: %f\n", last_five_avg);
						// last_five_avg =

						putText(image_processed, to_string(length), cvPoint(center_x, center_y), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
					}
					// printf("length:%f\tlfa:%f\n", length, last_five_avg);

					flux[f - START_FRAME] = last_five_avg;
					length_arr[f - START_FRAME] = length;
					speed[f-START_FRAME] = last_five_avg;
				}

				// printf("x:%ld\ty:%f\n", (f-START_FRAME)*10, 10*flux[f-START_FRAME]);

				// for (unsigned int i = 0; i < f-START_FRAME; i++)
				// {
				// 	circle(image_processed, Point((i)*3, (720 - 10*flux[i])), 2, Scalar(0, 0, 255), 2, 8);
				// 	circle(image_processed, Point(bbox_heights[i]*8, (720 - 10*flux[i])), 2, Scalar(255, 0, 0), 2, 8);
				// }
				// // circle(image_processed, Point( 200, 200 ), 32.0, Scalar( 0, 0, 255 ), 1, 8 );
				// circle(image_processed, Point(center_x, center_y), 2, Scalar(0, 255, 255), 2, 8);


				// imshow("flowwww", image_processed);
				// cvWaitKey(1);

				std::swap(prevgray, gray);

				imshow("traffic", image_processed);

				try
				{
					imwrite(image_file_processed_new, image_processed, compression_params);
				}
				catch (runtime_error& ex)
				{
					fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					return 1;
				}

				// Uncomment to wait for keypress when match is detected
				cvWaitKey(1);
			}
			else
			{
				// printf("here?\n");
				if (matched_frames == MINIMUM_FRAMES)
				{
					car_count--;
				}
				char image_file_processed[100];
				char image_file_processed_new[100];
				sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
				sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
				Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);

				if (((f - START_FRAME) % RECALCULATE_FRAMES) == 0)
				{
					double tc_speed = 140 - (car_count - prev_car_count) / DISTANCE / 2;
					string tc = "-1";

					if (tc_speed > 123.33)			{	tc = "A";	}
					else if (tc_speed > 106.66) 	{	tc = "B";	}
					else if (tc_speed > 90) 		{	tc = "C";	}
					else if (tc_speed > 73.33) 		{	tc = "D";	}
					else if (tc_speed > 56.66) 		{	tc = "E";	}
					else if (tc_speed > 40) 		{	tc = "F";	}
					else if (tc_speed > 23.33) 		{	tc = "7";	}
					else if (tc_speed > 6.66) 		{	tc = "8";	}
					else if (tc_speed > -10) 		{	tc = "9";	}
					else 							{	tc = "10";}

					traffic_class = "Traffic class: " + tc;
					prev_car_count = car_count;
				}

				// traffic_class = "Traffic class: N/A";
				current_car_speed = "Current Car Speed: N/A";
				string depth_string = "Current Car Depth: N/A";

				putText(image_processed, traffic_class, cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
				putText(image_processed, depth_string, cvPoint(10, 80), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
				putText(image_processed, current_car_speed, cvPoint(10, 110), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
				putText(image_processed, average_car_speed, cvPoint(10, 140), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);

				imshow("traffic", image_processed);

				try
				{
					// printf("WRITING IMAGE\n");
					imwrite(image_file_processed_new, image_processed, compression_params);
				}
				catch (runtime_error& ex)
				{
					fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
					return 1;
				}

				// Uncomment to wait for keypress when match is detected
				cvWaitKey(1);
				// }
			}
		}
		else
		{
			char image_file_processed[100];
			char image_file_processed_new[100];
			sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame0%04ld.jpeg_p.png", f);
			Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);

			if (((f - START_FRAME) % RECALCULATE_FRAMES) == 0)
			{
				double tc_speed = 140 - (car_count - prev_car_count) / DISTANCE / 2;
				string tc = "-1";

				if (tc_speed > 123.33)			{	tc = "A";	}
				else if (tc_speed > 106.66) 	{	tc = "B";	}
				else if (tc_speed > 90) 		{	tc = "C";	}
				else if (tc_speed > 73.33) 		{	tc = "D";	}
				else if (tc_speed > 56.66) 		{	tc = "E";	}
				else if (tc_speed > 40) 		{	tc = "F";	}
				else if (tc_speed > 23.33) 		{	tc = "7";	}
				else if (tc_speed > 6.66) 		{	tc = "8";	}
				else if (tc_speed > -10) 		{	tc = "9";	}
				else 							{	tc = "10";}

				traffic_class = "Traffic class: " + tc;
				prev_car_count = car_count;
			}

			// traffic_class = "Traffic class: N/A";
			current_car_speed = "Current Car Speed: N/A";
			string depth_string = "Current Car Depth: N/A";

			putText(image_processed, traffic_class, cvPoint(10, 30), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, depth_string, cvPoint(10, 80), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, current_car_speed, cvPoint(10, 110), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
			putText(image_processed, average_car_speed, cvPoint(10, 140), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);

			vector<int> compression_params;
			compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
			compression_params.push_back(9);

			imshow("traffic", image_processed);

			try
			{
				// printf("WRITING IMAGE 2\n");
				imwrite(image_file_processed_new, image_processed, compression_params);
			}
			catch (runtime_error& ex)
			{
				fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
				return 1;
			}

			// Uncomment to wait for keypress when match is detected
			cvWaitKey(1);
		}
	}

	ofstream log_file;
	log_file.open(LOG_FILE_NAME);

	log_file << "frame_number,flow,bbox_height,center_x,center_y,depth,speed\n";

	printf("%ld %ld %ld %ld", flux.size(), bbox_heights.size(), center_x_arr.size(), center_y_arr.size());

	for (unsigned int i = 0; i < flux.size(); i++)
	{
		// if (flux[i] != 0)
		// {
		log_file << i << "," << length_arr[i] << "," << bbox_heights[i] << "," << center_x_arr[i] << "," << center_y_arr[i] << "," << depth_arr[i] << "," << SPEED_FILE << endl;
		// }
	}

	return 0;
}