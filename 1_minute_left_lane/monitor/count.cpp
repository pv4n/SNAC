#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
	Mat image;

	int image_index = 2600;
	char image_file[100];
	sprintf(image_file, "../1_minute_left_lane/frame0%d.jpeg", image_index);
	image = imread(image_file, CV_LOAD_IMAGE_COLOR);

	cv::Mat image_mat(image);

	char frame_name[100];
	sprintf(frame_name, "Orignal Frame: %s", image_file);

	namedWindow(frame_name, WINDOW_AUTOSIZE);
	imshow(frame_name, image);


	// Read bounding box data
	ifstream bbox_file("car_boxes.txt");
	string line;


	getline(bbox_file, line);

	char frame[100];
	float xmin, ymin, xmax, ymax;
	sscanf(line.c_str(), "%s %*f %f %f %f %f", frame, &xmin, &ymin, &xmax, &ymax);
	printf("frame: %s\txmin: %f\t, ymin: %f\t, xmax: %f\t, ymax: %f\t\n", frame, xmin, ymin, xmax, ymax);



	int frame_number;
	sscanf(frame, "frame%d", &frame_number);
	printf("%d\n", frame_number);

	// std::vector<int>



	bbox_file.close();








	cv::Rect bound_box((int)xmin, (int)ymin, (int)(xmax-xmin), (int)(ymax-ymin));

	cv::Mat cropped = image_mat(bound_box);



	namedWindow( "Display window2", WINDOW_NORMAL );//mak Create a window for display.
	imshow( "Display window2", cropped );                   // Show our image inside it.

	waitKey(0);                                          // Wait for a keystroke in the window
	return 0;

// // You mention that you start with a CVMat* imagesource
// CVMat * imagesource;

// // Transform it into the C++ cv::Mat format
// cv::Mat image(imagesource);

// // Setup a rectangle to define your region of interest
// cv::Rect myROI(10, 10, 100, 100);

// // Crop the full image to that image contained by the rectangle myROI
// // Note that this doesn't copy the data
// cv::Mat croppedImage = image(myROI);

}