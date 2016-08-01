#include <cxcore.h>
#include <cv.h>
#include <fstream>
#include <highgui.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
// #include <opencv2/imgcodecs.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/contrib/contrib.hpp"

extern "C"
{
	#include "imgfeatures.h"
	#include "kdtree.h"
	#include "sift.h"
	#include "utils.h"
	#include "xform.h"
}

// the maximum number of keypoint NN candidates to check during BBF search
#define KDTREE_BBF_MAX_NN_CHKS 200

// threshold on squared ratio of distances between NN and 2nd NN
#define NN_SQ_DIST_RATIO_THR 0.49

#define MINIMUM_FRAMES 3
#define MINIMUM_FEATURES 10
#define XMAX_THRES 640
#define YMAX_THRES 650

#define START_FRAME 14
#define END_FRAME 82

#define BBOX_FILE "../data/speed_tracking_v3/bbox_files/car_h25.txt"
#define RAW_FRAMES_LOCATION "../data/speed_tracking_v3/frames/1_raw/h25"
#define PROCESSED_FRAMES_LOCATION "../data/speed_tracking_v3/frames/2_yolo/h25"
#define COUNTED_FRAMES_LOCATION "../data/speed_tracking_v3/frames/3_processed/h25"

using namespace cv;
using namespace std;

void peekline(ifstream &is, string &s)
{
	streampos sp = is.tellg();
	getline(is, s);
	is.seekg(sp);
}

double limit_value(double x, double a, double b)
{
	return ((x < a) ? a : ((x > b) ? b : x));
}

// static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color)
// {
// 	for(int y = 0; y < cflowmap.rows; y += step)
// 	{
// 		for(int x = 0; x < cflowmap.cols; x += step)
// 		{
// 			const Point2f& fxy = flow.at<Point2f>(y, x);
// 			line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
// 			circle(cflowmap, Point(x,y), 2, color, -1);
// 		}
// 	}
// }

int main(int argc, char** argv)
{
	Mat image;

	long int image_index = START_FRAME;
	long int frame_number = START_FRAME;

	// Read bounding box data
	ifstream bbox_file(BBOX_FILE);

	// Vector of vectors of vectors for bbox data
	vector<vector<vector<float>>> bboxes(END_FRAME - START_FRAME);

	float xmin, ymin, xmax, ymax;

	vector<double> flux(END_FRAME - START_FRAME);
	vector<double> bbox_heights(END_FRAME - START_FRAME);
	vector<double> speed(END_FRAME - START_FRAME);


	// Read bbox data from file
	do
	{
		// Check if frame number in bbox file matches frame index
		string line;
		peekline(bbox_file, line);
		sscanf(line.c_str(), "frame%ld", &frame_number);

		// If yes, then continue
		if (frame_number == image_index)
		{
			getline(bbox_file, line);
			vector<float> current_bbox(4);
			sscanf(line.c_str(), "%*s %*f %f %f %f %f", &xmin, &ymin, &xmax, &ymax);

			xmin = limit_value(xmin, 0, 1280);
			ymin = limit_value(xmin, 0, 720);
			xmax = limit_value(xmin, 0, 1280);
			ymax = limit_value(xmin, 0, 720);

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
	int prev_frames = 0;
	int car_count = 0;

	Mat flow, cflow, frame;
	Mat gray, prevgray, uflow;

	for (long int f=START_FRAME+1; f<END_FRAME; f++)
	{
		char image_file_zero[100];
		sprintf(image_file_zero, RAW_FRAMES_LOCATION"/frame%05ld.jpeg", f-1);

		// printf("%s\n", image_file_zero);
		Mat image_zero = imread(image_file_zero, CV_LOAD_IMAGE_COLOR);

		if(bboxes[f-START_FRAME-1].empty())
		{
			continue;
		}

		printf("\n\n%f\n\n", bboxes[f-START_FRAME-1][0][0]);

		cv::Rect bound_box_zero(bboxes[f-START_FRAME-1][0][0],
								bboxes[f-START_FRAME-1][0][1],
								bboxes[f-START_FRAME-1][0][2] - bboxes[f-START_FRAME-1][0][0],
								bboxes[f-START_FRAME-1][0][3] - bboxes[f-START_FRAME-1][0][1]);

		// imshow("test", bound_box_zero);
		// cvWaitKey(0);

		cv::Mat cropped_zero = image_zero(bound_box_zero);

		IplImage img1_image = cropped_zero;
		IplImage* img1 = &img1_image;


		char image_file_one[100];
		sprintf(image_file_one, RAW_FRAMES_LOCATION"/frame%05ld.jpeg", f);
		Mat image_one = imread(image_file_one, CV_LOAD_IMAGE_COLOR);

		if(bboxes[f-START_FRAME].empty())
		{
			continue;
		}
		cv::Rect bound_box_one(bboxes[f-START_FRAME][0][0],
								bboxes[f-START_FRAME][0][1],
								bboxes[f-START_FRAME][0][2] - bboxes[f-START_FRAME][0][0],
								bboxes[f-START_FRAME][0][3] - bboxes[f-START_FRAME][0][1]);
		cv::Mat cropped_one = image_one(bound_box_one);
		IplImage img2_image = cropped_one;
		IplImage* img2 = &img2_image;


		// //-- 2. Call the constructor for StereoBM
		// int ndisparities = 16*5;   /**< Range of disparity */
		// int SADWindowSize = 21; /**< Size of the block window. Must be odd */

		// Ptr<StereoBM> sbm = StereoBM::create( ndisparities, SADWindowSize );

		// //-- 3. Calculate the disparity image
		// sbm->compute( image_zero, image_one, imgDisparity16S );

		// //-- Check its extreme values
		// double minVal; double maxVal;

		// minMaxLoc( imgDisparity16S, &minVal, &maxVal );

		// printf("Min disp: %f Max value: %f \n", minVal, maxVal);

		// Mat imgDisparity16S = Mat( image_zero.rows, image_zero.cols, CV_16S );
		// Mat imgDisparity8U = Mat( image_zero.rows, image_zero.cols, CV_8UC1 );

		// //-- 4. Display it as a CV_8UC1 image
		// imgDisparity16S.convertTo( imgDisparity8U, CV_8UC1, 255/(maxVal - minVal));

		// namedWindow( "windowDisparity", WINDOW_NORMAL );
		// imshow( "windowDisparity", imgDisparity8U );


		// Mat disp, disp8, g1, g2;

		// //char* method = argv[3];
		// char* method = "SGBM";

		//img1 = imread(argv[1]);
		//img2 = imread(argv[2]);
		// img1 = imread("leftImage.jpg");
		// img2 = imread("rightImage.jpg");

		// cvtColor(image_zero, g1, CV_BGR2GRAY);
		// cvtColor(image_one, g2, CV_BGR2GRAY);

		// if (!(strcmp(method, "BM")))
		// {
		//     StereoBM sbm;
		//     sbm.state->SADWindowSize = 9;
		//     sbm.state->numberOfDisparities = 112;
		//     sbm.state->preFilterSize = 5;
		//     sbm.state->preFilterCap = 61;
		//     sbm.state->minDisparity = -39;
		//     sbm.state->textureThreshold = 507;
		//     sbm.state->uniquenessRatio = 0;
		//     sbm.state->speckleWindowSize = 0;
		//     sbm.state->speckleRange = 8;
		//     sbm.state->disp12MaxDiff = 1;
		//     sbm(g1, g2, disp);
		// }
		// else if (!(strcmp(method, "SGBM")))
		// {
		//     StereoSGBM sbm;
		//     sbm.SADWindowSize = 3;
		//     sbm.numberOfDisparities = 144;
		//     sbm.preFilterCap = 63;
		//     sbm.minDisparity = -39;
		//     sbm.uniquenessRatio = 10;
		//     sbm.speckleWindowSize = 100;
		//     sbm.speckleRange = 32;
		//     sbm.disp12MaxDiff = 1;
		//     sbm.fullDP = false;
		//     sbm.P1 = 216;
		//     sbm.P2 = 864;
		//     sbm(g1, g2, disp);
		// }


		// normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);

		// imshow("left", image_zero);
		// imshow("right", image_one);
		// imshow("disp", disp8);

		// waitKey(0);




		// IplImage *stacked;
		struct feature* feat1, * feat2, * feat;
		struct feature** nbrs;
		struct kd_node* kd_root;
		CvPoint pt1, pt2;
		double d0, d1;
		int n1, n2, k, i, m = 0;



		// stacked = stack_imgs(img1, img2);



		// fprintf( stderr, "Finding features in %s...\n", "CROPPED ONE");
		n1 = sift_features(img1, &feat1);
		// fprintf( stderr, "Finding features in %s...\n", "CROPPED TWO" );
		n2 = sift_features(img2, &feat2);
		// fprintf( stderr, "Building kd tree...\n" );
		kd_root = kdtree_build(feat2, n2);

printf("\n\nsorted\n\n");

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
							// cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
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

		// printf("%ld\n", f);

		if (m > MINIMUM_FEATURES)
		{
			matched_frames++;
		}
		else
		{
			if (matched_frames > 3)
			{
				prev_frames = matched_frames;
			}
			matched_frames = 0;
		}

		if (matched_frames >= MINIMUM_FRAMES)
		{
			if (matched_frames == MINIMUM_FRAMES)
			{
				printf("\n\n---------------------------------\n");
				printf("Number of Frames of prev car: %d\n", prev_frames);
				printf("---------------------------------\n\n\n");
				car_count++;
			}

			if ((bboxes[f-START_FRAME][0][2] < XMAX_THRES) && (bboxes[f-START_FRAME][0][3] < YMAX_THRES))
			{

				int center_x = (bboxes[f-START_FRAME][0][0] + bboxes[f-START_FRAME][0][2]) / 2;
				int center_y = (bboxes[f-START_FRAME][0][1] + bboxes[f-START_FRAME][0][3]) / 2;


				printf("Found unique car #%d\n", car_count);

				char image_file_processed[100];
				char image_file_processed_new[100];
				sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame%05ld.jpeg_p.png", f);
				sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame%05ld.jpeg_p.png", f);
				Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);
				putText(image_processed, to_string(car_count), cvPoint(bboxes[f-START_FRAME][0][2], bboxes[f-START_FRAME][0][3]), FONT_HERSHEY_SIMPLEX, 2, cvScalar(0, 0, 255), 5, CV_AA);

				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);

				char input[100];
				sprintf(input, RAW_FRAMES_LOCATION"/frame%05ld.jpeg", f);
				Mat current_image = imread(input, CV_LOAD_IMAGE_COLOR);
				cvtColor(current_image, gray, COLOR_BGR2GRAY);

				bbox_heights[f-START_FRAME] = bboxes[f-START_FRAME][0][3] - bboxes[f-START_FRAME][0][1];

				if( !prevgray.empty() )
				{
					calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
					cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
					uflow.copyTo(flow);
					// drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
					// imshow("flow", cflow);
					const Point2f& fxy = flow.at<Point2f>(center_y, center_x);
					float length = sqrt(pow(fxy.x, 2) + pow(fxy.y, 2));
					printf("Flow at (%d, %d):\t%f\n\n", center_x, center_y, length);

					// double ratio = 0;
					double last_five_avg = 0;

					if (f-START_FRAME > 5)
					{
						double sum = 0;
						double bbox_box_sum = 0;
						for (int i=0; i < 5; i++)
						{
							sum += bbox_heights[f-START_FRAME - i];
							bbox_box_sum += bbox_heights[f-START_FRAME - i];
						}

						//ratio = 35639.6340611629 * length / (bbox_box_sum * bbox_box_sum);
						last_five_avg = sum / 5;

						// last_five_avg =

						putText(image_processed, to_string(length), cvPoint(center_x, center_y), FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 255), 2, CV_AA);
					}

					flux[f - START_FRAME] = length;
					speed[f-START_FRAME] = last_five_avg;
				}

				printf("x:%ld\ty:%f\n", (f-START_FRAME)*10, 10*flux[f-START_FRAME]);


				for (unsigned int i = 0; i < f-START_FRAME; i++)
				{
					circle(image_processed, Point((i)*3, (720 - 10*flux[i])), 2, Scalar(0, 0, 255), 2, 8);
					circle(image_processed, Point(bbox_heights[i]*8, (720 - 10*flux[i])), 2, Scalar(255, 0, 0), 2, 8);
				}
				// circle(image_processed, Point( 200, 200 ), 32.0, Scalar( 0, 0, 255 ), 1, 8 );

				imshow("flowwww", image_processed);
				cvWaitKey(1);

				std::swap(prevgray, gray);

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
				int c = cvWaitKey(30);
				if (c > 0)
				{
					break;
				}
			}
			else if (matched_frames == MINIMUM_FRAMES)
			{
				car_count--;
				char image_file_processed[100];
				char image_file_processed_new[100];
				sprintf(image_file_processed, PROCESSED_FRAMES_LOCATION"/frame%05ld.jpeg_p.png", f);
				sprintf(image_file_processed_new, COUNTED_FRAMES_LOCATION"/frame%05ld.jpeg_p.png", f);
				vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);
				Mat image_processed = imread(image_file_processed, CV_LOAD_IMAGE_COLOR);
				imwrite(image_file_processed_new, image_processed, compression_params);
			}
		}
	}

	printf("\n-----------------FLUX-----------------\n");
	for (auto i = flux.begin(); i != flux.end(); ++i)
		std::cout << *i << '\n';

	printf("\n------------BBOX HEIGHTS--------------\n");
	for (auto i = bbox_heights.begin(); i != bbox_heights.end(); ++i)
		std::cout << *i << '\n';

	printf("\n------------SPEED--------------\n");
	for (auto i = speed.begin(); i != speed.end(); ++i)
		std::cout << *i << '\n';

	cvWaitKey(0);

	// namedWindow("plot", WINDOW_NORMAL);
	// Mat plot(720, 1280, CV_8UC4);

	// for (unsigned int i = 0; i < flux.size(); i++)
	// {
	// 	circle(image_processed, Point2f(i*10, flux[i]), 2, Scalar(0, 255, 0));
	// }

	// imshow("flowwww", image_processed);
	// cvWaitKey(0);

	return 0;
}