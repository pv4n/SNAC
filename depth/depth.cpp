#include <cxcore.h>
#include <cv.h>
#include <fstream>
#include <highgui.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <vector>

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

#define START_FRAME 192
#define END_FRAME 315

#define BBOX_FILE "../data/speed_tracking/bbox_files/car_45.txt"
#define RAW_FRAMES_LOCATION "../data/speed_tracking/raw_frames/45"
#define PROCESSED_FRAMES_LOCATION "../data/speed_tracking/processed_frames/45"
#define COUNTED_FRAMES_LOCATION "../data/speed_tracking/counted_frames/45"

using namespace cv;
using namespace std;

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

	// Read bounding box data
	ifstream bbox_file(BBOX_FILE);

	// Vector of vectors of vectors for bbox data
	vector<vector<vector<float>>> bboxes(END_FRAME - START_FRAME);

	float xmin, ymin, xmax, ymax;

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

	for (long int f=START_FRAME+1; f<END_FRAME; f++)
	{
		char image_file_zero[100];
		sprintf(image_file_zero, RAW_FRAMES_LOCATION"/frame%05ld.jpeg", f-1);

		printf("hey%s\n", image_file_zero);
		Mat image_zero = imread(image_file_zero, CV_LOAD_IMAGE_COLOR);

		imshow("test", image_zero);
		cvWaitKey(0);

		if(bboxes[START_FRAME-1].empty())
		{
			continue;
		}
		cv::Rect bound_box_zero(bboxes[START_FRAME-1][0][0],
								bboxes[START_FRAME-1][0][1],
								bboxes[START_FRAME-1][0][2] - bboxes[START_FRAME-1][0][0],
								bboxes[START_FRAME-1][0][3] - bboxes[START_FRAME-1][0][1]);
		cv::Mat cropped_zero = image_zero(bound_box_zero);
		IplImage img1_image = cropped_zero;
		IplImage* img1 = &img1_image;

		puts("here\n");


		char image_file_one[100];
		sprintf(image_file_one, RAW_FRAMES_LOCATION"/raw_frames/frame%05ld.jpeg", f);
		Mat image_one = imread(image_file_one, CV_LOAD_IMAGE_COLOR);

		if(bboxes[START_FRAME].empty())
		{
			continue;
		}
		cv::Rect bound_box_one(bboxes[START_FRAME][0][0],
								bboxes[START_FRAME][0][1],
								bboxes[START_FRAME][0][2] - bboxes[START_FRAME][0][0],
								bboxes[START_FRAME][0][3] - bboxes[START_FRAME][0][1]);
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

		printf("%ld\n", f);

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
				// cvWaitKey(0);
			}
			else if (matched_frames == MINIMUM_FRAMES)
			{
				car_count--;
			}
		}
	}
	return 0;
}