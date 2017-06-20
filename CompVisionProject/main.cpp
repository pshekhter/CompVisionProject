/*
	This project is being developed for the CMSC-498 - Senior Seminar Fall 2017 semester at Chestnut Hill College in Philadelphia, PA.
	The purpose is to test the accuracy and efficiency of the Canny edge detector given different filters, using the Laplacian of Gaussian and Hough
	pseudo-wavelet as benchmarks.
	- Pavel Shekhter
*/

#include <opencv2/opencv.hpp>
#include <opencv_modules.hpp>
#include <string>

struct VIDEODATA {
	cv::VideoCapture vc;
	cv::Size s;
	cv::VideoWriter w;
} vd;

cv::Mat currentFrameColor;
cv::Mat currentFrameGry;

/*
	Moves on to the next frame.
	- Pavel Shekhter
*/
void on_trackbar(int, void*) {

}

/*
	Reads the video file into the frame buffer.
	- Pavel Shekhter
*/
VIDEODATA readVideoFile(std::string filename) {

	cv::VideoCapture capture(filename);
	double fps = capture.get(cv::CAP_PROP_FPS);
	cv::Size size(
		(int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
		(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	cv::VideoWriter writer;
	writer.open(filename, CV_FOURCC('M', 'J', 'P', 'G'), fps, size);
	VIDEODATA vd;
	vd.vc = capture;
	vd.s = size;
	vd.w = writer;
	return vd;

}

int main(int argc, char** argv) {

	std::string filename;
	
	std::cout << "Enter video file name: " << std::endl;
	std::cin >> filename;

	VIDEODATA vd = readVideoFile(filename);

	return 0;
}
