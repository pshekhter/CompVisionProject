#include <opencv2/opencv.hpp>
#include <opencv_modules.hpp>

int main(int argc, char** argv) {
	
	cv::VideoCapture cap(0); // open default camera
	if (!cap.isOpened()) {
		return -1;
	}

	cv::Mat original, edges, gray;

	cv::namedWindow("Original", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Edges", cv::WINDOW_AUTOSIZE);

	for (;;) {
		cap >> original; // Get new frame from camera
		cv::imshow("Original", original);
		cv::cvtColor(original, gray, cv::COLOR_BGR2GRAY);
		cv::Scharr(gray, edges, -1, 1, 0, 0, 0, 4);
		cv::Canny(gray, edges, 35, 35 * 2);
		cv::imshow("Edges", edges);
		char c = cv::waitKey(10);
		if (c == 27) break;
	}
	cap.release();
	return 0;
}
