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
#include <functional>
#include <vector>
#include <boost/filesystem.hpp>
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
		 case -3: {
			 file << "Unable to output to image." << std::endl;
			 break;
		 }
	 }
 }

/*
	Parses the arguments from the command line.
	- Pavel Shekhter
*/
 int parseArguments(int argc, char * argv, std::ofstream &file, bool &retflag)
 {
	 retflag = true;
	 std::string imagefile(argv);

	 if (!imagefile.empty()) {
		 IMAGEDATA id = readImageData(imagefile);

		 if (!id.currentFrameColor.data) {
			 std::cout << "Can't open file!" << std::endl;
			 appendErrorMessage(file, -1);
			 return -1;
		 }
	 }

	 file << "Processing File: " << argv << std::endl;

	 retflag = false;
	 return {};
 }

 /*
	Performs a Canny edge detector using Gaussian blur.
	- Pavel Shekhter
 */
 void gaussianCanny(std::ofstream &file, char * argv, cv::Mat &mat) {
	 file << "Starting Canny w/Gaussian Blur. Initial Time: ";
	 double initGCTime = (cv::getTickCount()) / (cv::getTickFrequency());
	 file << initGCTime * 1000 << " ms" << std::endl;
	 cv::blur(id.currentFrameGry, id.cannyGaussianDetectedEdges, cv::Size(3, 3));
	 cv::Canny(id.cannyGaussianDetectedEdges, id.cannyGaussianDetectedEdges, id.lowThresh, id.lowThresh * id.ratio, id.kernel);
	 cv::Mat dst;
	 dst = cv::Scalar::all(0);
	 id.currentFrameGry.copyTo(dst, id.cannyGaussianDetectedEdges);
	 file << "Canny w/Gaussian Blur finished. Final Time: ";
	 double finalGCTime = (cv::getTickCount()) / (cv::getTickFrequency());
	 file << finalGCTime * 1000 << " ms" << std::endl;
	 file << "Canny w/Gaussian Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
	 mat = dst;
}

 int main(int argc, char* argv[]) {
	 std::ofstream file;

	if (!(argc > 1)) {
		std::cout << "Usage: CompVisionProject imageToLoad" << std::endl;
		appendErrorMessage(std::cout, -2);
		return -2;
	}

	std::string report;
	
	std::cout << "Enter report file name: " << std::endl;
	getline(std::cin, report);

	file.open(report);

	setUpFile(file, report);

	for (int i = 1; i < argc; i++) {
		bool retflag;
		int retval = parseArguments(argc, argv[i], file, retflag);
		if (retflag) return retval;

		cv::namedWindow("Computer Vision Demo", CV_WINDOW_AUTOSIZE);
		cv::imshow("Computer Vision Demo", id.currentFrameColor);
		cv::cvtColor(id.currentFrameColor, id.currentFrameGry, cv::COLOR_BGR2GRAY);

		cv::Mat gaussCannyDet;
		bool isGCDone = false;
		std::thread gaussCanny(&gaussianCanny, std::ref(file), argv[i], std::ref(gaussCannyDet));
		if (gaussCanny.joinable()) {
			gaussCanny.join();
			isGCDone = true;
		}

		if (!gaussCannyDet.empty() || isGCDone == true) {
			std::vector<int> comp_params;
			comp_params.push_back(CV_IMWRITE_JPEG_QUALITY);
			comp_params.push_back(100);
			std::string im = argv[i];
			boost::filesystem::path image_path(im);
			if (boost::filesystem::exists(image_path)) {
				std::string imp = image_path.filename().generic_string();
				try {
					bool save = cv::imwrite("gaussian_" + imp, gaussCannyDet, comp_params);
				}
				catch (std::runtime_error& e) {
					appendErrorMessage(file, -3);
					fprintf(stderr, "Unable to write file due to: %s\n", e.what());
				}
			}
		}

		if (!gaussCannyDet.empty()) {
			cv::namedWindow("Canny: Gaussian");
			cv::imshow("Canny: Gaussian", gaussCannyDet);
		}
		
		cv::waitKey(0);

	}

	file.close();

	return 0;
}
