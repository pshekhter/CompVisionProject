/*
	This project is being developed for the CMSC-498 - Senior Seminar Fall 2017 semester at Chestnut Hill College in Philadelphia, PA.
	The purpose is to test the accuracy and efficiency of the Canny edge detector given different filters, using the Laplacian of Gaussian and Hough
	pseudo-wavelet as benchmarks.
	- Pavel Shekhter
*/

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv_modules.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <fstream>
#include <thread>
#include "main.h"

struct IMAGEDATA {
	cv::Mat currentFrameColor;
	cv::Mat currentFrameGry;
	cv::Mat cannyGaussianDetectedEdges;
	int lowThresh;
	int ratio = 3;
	int kernel = 3;
} id;



/*
	Reads the video file into the frame buffer.
	- Pavel Shekhter
*/
 IMAGEDATA readImageData(std::string imagefile) {

	 cv::Mat image = cv::imread(imagefile, CV_LOAD_IMAGE_COLOR);
	 id.currentFrameColor = image;

	return id;
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

 /*
	Appends an error message to the report.
	- Pavel Shekhter
 */
 void appendErrorMessage(std::ostream &file, int errorCode) {
	 file << "There was an error in processing. Error Code: " << errorCode << std::endl;
	 file << "This means: ";
	 switch (errorCode) {
		 case -1: {
			file << "Error loading file." << std::endl;
			break;
		}
		 case -2: {
			 file << "Improper loading." << std::endl;
			 break;
		 }
	 }
 }

/*
	Parses the arguments from the command line.
	- Pavel Shekhter
*/
 int parseArguments(int argc, char ** argv, std::ofstream &file, bool &retflag)
 {
	 retflag = true;
	 std::string keys = "{@image         |          | image to scan}";

	 cv::CommandLineParser parser(argc, argv, keys);
	 std::string imagefile = parser.get<std::string>("@image");

	 if (!imagefile.empty()) {
		 IMAGEDATA id = readImageData(imagefile);

		 if (!id.currentFrameColor.data) {
			 std::cout << "Can't open file!" << std::endl;
			 appendErrorMessage(file, -1);
			 return -1;
		 }
	 }
	 retflag = false;
	 return {};
 }

 /*
	Performs a Canny edge detector using Gaussian blur.
	- Pavel Shekhter
 */
 cv::Mat gaussianCanny() {
	 cv::blur(id.currentFrameGry, id.currentFrameGry, cv::Size(3, 3));
	 cv::Canny(id.cannyGaussianDetectedEdges, id.cannyGaussianDetectedEdges, id.lowThresh, id.lowThresh * id.ratio, id.kernel);
	 cv::Mat dst;
	 dst = cv::Scalar::all(0);
	 id.currentFrameColor.copyTo(dst, id.cannyGaussianDetectedEdges);
	 return dst;
}

 int main(int argc, char** argv) {
	 std::ofstream file;

	if (argc != 2) {
		std::cout << "Usage: CompVisionProject imageToLoad" << std::endl;
		appendErrorMessage(std::cout, -2);
		return -2;
	}

	std::string report;
	
	std::cout << "Enter report file name: " << std::endl;
	getline(std::cin, report);

	file.open(report);

	setUpFile(file, report);

	bool retflag;
	int retval = parseArguments(argc, argv, file, retflag);
	if (retflag) return retval;

	cv::namedWindow("Computer Vision Demo", CV_WINDOW_AUTOSIZE);
	cv::imshow("Computer Vision Demo", id.currentFrameColor);
	cv::cvtColor(id.currentFrameColor, id.currentFrameGry, cv::COLOR_BGR2GRAY);

	cv::Mat det = gaussianCanny();
	cv::namedWindow("Gaussian", 1);
	cv::imshow("Gaussian", det);

	cv::waitKey(0);

	file.close();

	return 0;
}
