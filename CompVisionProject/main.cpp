/*
	This project is being developed for the CMSC-498 - Senior Seminar Fall 2017 semester at Chestnut Hill College in Philadelphia, PA.
	The purpose is to test the accuracy and efficiency of the Canny edge detector given different filters, using the Laplacian of Gaussian and Hough
	pseudo-wavelet as benchmarks.
	- Pavel Shekhter
*/

#include <opencv2/opencv.hpp>
#include <opencv_modules.hpp>
#include <string>
#include <fstream>

struct VIDEODATA {
	cv::VideoCapture vc;
	cv::Size s;
} vd;

cv::Mat currentFrameColor;
cv::Mat currentFrameGry;
std::string report;


/*
	Reads the video file into the frame buffer.
	- Pavel Shekhter
*/
 VIDEODATA readVideoData() {

	cv::VideoCapture capture(0);
	if (!capture.isOpened()) {
		std::cout << "Failed to open webcam!" << std::endl;
	}
	double fps = capture.get(cv::CAP_PROP_FPS);
	cv::Size size(
		(int)capture.get(cv::CAP_PROP_FRAME_WIDTH),
		(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
	VIDEODATA vd;
	vd.vc = capture;
	vd.s = size;

	return vd;
}

 /*
	Sets up the report.
	- Pavel Shekhter
 */
 void setUpFile(std::ofstream &file, std::string &report_name) {
	 file << "Edge Detection Analysis Data: " << std::endl;
	 file << "Report file name: " << report_name << std::endl;
	 file << "\n\n";
 }

int main(int argc, char** argv) {

	std::ofstream file;
	
	std::cout << "Enter report file name: " << std::endl;
	getline(std::cin, report);

	file.open(report);

	setUpFile(file, report);

	VIDEODATA vd = readVideoData();

	file.close();

	return 0;
}
