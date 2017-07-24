#define _USE_MATH_DEFINES

/*
    This project is being developed for the CMSC-498 - Senior Seminar Fall 2017 semester at Chestnut Hill College in Philadelphia, PA.
    The purpose is to test the accuracy and efficiency of the Canny, Sobel, and Scharr edge detectors, using the Laplacian of Gaussian and Hough
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
#include <cmath>
#include "main.h"

struct IMAGEDATA {
    cv::Mat currentFrameColor;
    cv::Mat currentFrameGry;
    cv::Mat cannyGaussianDetectedEdges;
    cv::Mat cannyNormalizedDetectedEdges;
    cv::Mat laplaceDest;
    cv::Mat sobelGrad;
    cv::Mat sobelXGrad;
    cv::Mat sobelYGrad;
    cv::Mat sobelAbsXGrad;
    cv::Mat sobelAbsYGrad;
    cv::Mat gaborDest;
    std::vector<cv::Mat> gaborKernels;
    cv::Mat gaborSrc_f;
    int canny_lowThresh;
    int canny_Ratio = 3;
    int canny_Kernel = 3;
    int laplace_kernel = 3;
    int laplace_scale = 1;
    int laplace_delta = 0;
    int laplace_ddepth = CV_16S;
    int sobel_scale = 1;
    int sobel_delta = 0;
    int sobel_ddepth = CV_16S;
    int gaborKernelSize = 31;
    double gaborSig = 4.0, gaborTh = M_PI / 16, gaborLm = 10.0, gaborGm = 0.5, gaborPs = 0;
} id;



/*
    Reads the video file into the frame buffer.
    - Pavel Shekhter
*/
IMAGEDATA readImageData (std::string imagefile) {

    cv::Mat image = cv::imread (imagefile, CV_LOAD_IMAGE_COLOR);
    id.currentFrameColor = image;

    return id;
}

/*
   Sets up the report.
   - Pavel Shekhter
*/
void setUpFile (std::ofstream &file, std::string &report_name, std::ofstream &csv) {
    file << "Edge Detection Analysis Data: " << std::endl;
    file << "Report file name: " << report_name << std::endl;
    file << "\n\n";

    csv << "Trial #, Laplacian w/ Gaussian Blur, Laplacian w/ Normalized Box Filter, Laplacian w/ Box Filter, ";
    csv << "Canny w/ Gaussian Blur, Canny w/ Normalized Box Filter, Canny w/ Box Filter, ";
    csv << "Sobel w/Gaussian Blur, Sobel w/ Normalized Box Filter, Sobel w/ Box Filter, Gabor filter-based edge detector w/ no additional filtering\n";
}

/*
   Appends an error message to the report.
   - Pavel Shekhter
*/
void appendErrorMessage (std::ostream &file, int errorCode) {
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
int parseArguments (int argc, char * argv, std::ofstream &file, bool &retflag) {
    retflag = true;
    std::string imagefile (argv);

    if (!imagefile.empty ()) {
        IMAGEDATA id = readImageData (imagefile);

        if (!id.currentFrameColor.data) {
            std::cout << "Can't open file!" << std::endl;
            appendErrorMessage (file, -1);
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
void gaussianCanny (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Canny w/Gaussian Blur. Initial Time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::GaussianBlur (id.currentFrameGry, id.cannyGaussianDetectedEdges, cv::Size (3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::Canny (id.cannyGaussianDetectedEdges, id.cannyGaussianDetectedEdges, id.canny_lowThresh, id.canny_lowThresh * id.canny_Ratio, id.canny_Kernel);
    cv::Mat dst;
    dst = cv::Scalar::all (0);
    id.currentFrameGry.copyTo (dst, id.cannyGaussianDetectedEdges);
    file << "Canny w/Gaussian Blur finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Canny w/Gaussian Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = dst;
}

/*
Performs a Canny edge detector using normalized box blur.
- Pavel Shekhter
*/
void normalizedCanny (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Canny w/Normalized Box Blur. Initial Time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::blur (id.currentFrameGry, id.cannyNormalizedDetectedEdges, cv::Size (3, 3));
    cv::Canny (id.cannyNormalizedDetectedEdges, id.cannyNormalizedDetectedEdges, id.canny_lowThresh, id.canny_lowThresh * id.canny_Ratio, id.canny_Kernel);
    cv::Mat dst;
    dst = cv::Scalar::all (0);
    id.currentFrameGry.copyTo (dst, id.cannyNormalizedDetectedEdges);
    file << "Canny w/Normalized Box Blur finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Canny w/Normalized Box Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = dst;
}

/*
Performs a Canny edge detector using box filter.
- Pavel Shekhter
*/
void boxCanny (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Canny w/Box Filter. Initial Time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::boxFilter (id.currentFrameGry, id.cannyNormalizedDetectedEdges, -1, cv::Size (3, 3), cv::Point (-1, -1), true, cv::BORDER_DEFAULT);
    cv::Canny (id.cannyNormalizedDetectedEdges, id.cannyNormalizedDetectedEdges, id.canny_lowThresh, id.canny_lowThresh * id.canny_Ratio, id.canny_Kernel);
    cv::Mat dst;
    dst = cv::Scalar::all (0);
    id.currentFrameGry.copyTo (dst, id.cannyNormalizedDetectedEdges);
    file << "Canny w/Box Filter finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Canny w/Box Filter took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = dst;
}

/*
Performs a Laplacian edge detector using Gausian filter.
- Pavel Shekhter
 */
void gausianLaplace (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Laplacian w/ Gaussian Blur. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::GaussianBlur (id.currentFrameColor, id.currentFrameColor, cv::Size (3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);
    cv::Mat abs_dst;
    cv::Laplacian (id.currentFrameGry, mat, id.laplace_ddepth, id.laplace_kernel, id.laplace_scale, id.laplace_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (mat, abs_dst);
    mat = abs_dst;
    id.laplaceDest = abs_dst;
    file << "Laplacian w/ Gaussian Blur finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Laplacian w/ Gaussian Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";

}

/*
Performs a Laplacian edge detector using normalized box blur.
- Pavel Shekhter
*/
void normalizedLaplace (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Laplacian w/ Normalized Box Blur. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::blur (id.currentFrameColor, id.currentFrameColor, cv::Size (3, 3));
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);
    cv::Mat abs_dst;
    cv::Laplacian (id.currentFrameGry, mat, id.laplace_ddepth, id.laplace_kernel, id.laplace_scale, id.laplace_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (mat, abs_dst);
    mat = abs_dst;
    id.laplaceDest = abs_dst;
    file << "Laplacian w/ Normalized Box Blur finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Laplacian w/ Normalized Box Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";

}

/*
Performs a Laplacian edge detector using box filter.
- Pavel Shekhter
*/
void boxLaplace (std::ofstream &file, char * argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Laplacian w/ Box filter. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::boxFilter (id.currentFrameColor, id.currentFrameColor, -1, cv::Size (3, 3), cv::Point (-1, -1), true, cv::BORDER_DEFAULT);
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);
    cv::Mat abs_dst;
    cv::Laplacian (id.currentFrameGry, mat, id.laplace_ddepth, id.laplace_kernel, id.laplace_scale, id.laplace_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (mat, abs_dst);
    mat = abs_dst;
    id.laplaceDest = abs_dst;
    file << "Laplacian w/ Box Filter finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Laplacian w/ Box Filter Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";

}

/*
Perform Sobel edge detection using Gaussian blur
 */
void gaussianSobel (std::ofstream &file, char *argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Sobel w/ Gaussian Blur. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::GaussianBlur (id.currentFrameColor, id.currentFrameColor, cv::Size (3, 3), 0, 0, cv::BORDER_DEFAULT);
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);

    // Perform Sobel on X-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelXGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelXGrad, id.sobelAbsXGrad);

    // Perform Sobel on Y-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelYGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelYGrad, id.sobelAbsYGrad);

    // Add Gradients
    cv::addWeighted (id.sobelAbsXGrad, 0.5, id.sobelAbsYGrad, 0.5, 0, id.sobelGrad);
    file << "Sobel w/ Gaussian Blur finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Sobel w/ Gaussian Blur took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = id.sobelGrad;

}

/*
Perform Sobel edge detection using Normalized Box Filter
*/
void normalizedSobel (std::ofstream &file, char *argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Sobel w/ Normalized Box Filter. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::blur (id.currentFrameColor, id.currentFrameColor, cv::Size (3, 3));
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);

    // Perform Sobel on X-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelXGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelXGrad, id.sobelAbsXGrad);

    // Perform Sobel on Y-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelYGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelYGrad, id.sobelAbsYGrad);

    // Add Gradients
    cv::addWeighted (id.sobelAbsXGrad, 0.5, id.sobelAbsYGrad, 0.5, 0, id.sobelGrad);
    file << "Sobel w/ Normalized Box Filter finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Sobel w/ Normalized Box Filter took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = id.sobelGrad;
}

/*
Perform Sobel edge detection using Normalized Box Filter
*/
void boxSobel (std::ofstream &file, char *argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Sobel w/ Box Filter. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    cv::boxFilter (id.currentFrameColor, id.currentFrameColor, -1, cv::Size (3, 3), cv::Point (-1, -1), true, cv::BORDER_DEFAULT);
    cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_RGB2GRAY);

    // Perform Sobel on X-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelXGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelXGrad, id.sobelAbsXGrad);

    // Perform Sobel on Y-Gradient
    cv::Sobel (id.currentFrameGry, id.sobelYGrad, id.sobel_ddepth, 1, 0, 3, id.sobel_scale, id.sobel_delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs (id.sobelYGrad, id.sobelAbsYGrad);

    // Add Gradients
    cv::addWeighted (id.sobelAbsXGrad, 0.5, id.sobelAbsYGrad, 0.5, 0, id.sobelGrad);
    file << "Sobel w/ Box Filter finished. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Sobel w/ Box Filter took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";
    mat = id.sobelGrad;
}

/*
Perform a Gabor filter-based edge detector with no additional filtering
*/
void gabor (std::ofstream &file, char *argv, cv::Mat &mat, std::ofstream &csv) {
    file << "Starting Gabor filter-based edge detector w/ no additional filtering. Initial time: ";
    double initGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << initGCTime * 1000 << " ms" << std::endl;
    mat = id.currentFrameGry;

    // Create a vector of kernels and filter
    for (int i = 0; i < (M_PI / 16); i += (M_PI / 2)) {
        cv::Mat kern;
        kern = cv::getGaborKernel (cv::Size (id.gaborKernelSize, id.gaborKernelSize), id.gaborSig, id.gaborTh, id.gaborLm, id.gaborGm, id.gaborPs, CV_32F);
        id.gaborKernels.push_back (kern);
    }

    id.gaborDest = mat;

    for (int i = 0; i < id.gaborKernels.size (); ++i) {
        cv::filter2D (id.gaborDest, id.gaborDest, CV_32F, id.gaborKernels.at (i));
        cv::Mat accum;
        accum = cv::Mat::zeros (id.gaborDest.size (), id.gaborDest.type ());
        cv::normalize (id.gaborDest, id.gaborDest, 0, 255, cv::NORM_MINMAX);



        id.gaborDest.convertTo (id.gaborDest, CV_8U, 1, 0); // Shift into proper 1..255 display range
    }

    file << "Gabor filter-based edge detector w/ no additional filtering completed. Final Time: ";
    double finalGCTime = (cv::getTickCount ()) / (cv::getTickFrequency ());
    file << finalGCTime * 1000 << " ms" << std::endl;
    file << "Gabor filter-based edge detector w/ no additional filtering took " << ((finalGCTime - initGCTime) * 1000) << " ms to complete." << std::endl;
    csv << (finalGCTime - initGCTime) * 1000 << ", ";

    std::cerr << id.gaborDest (cv::Rect (30, 30, 10, 10)) << std::endl; // Peek into data
    std::cerr << mat (cv::Rect (30, 30, 10, 10)) << std::endl; // Peek into data
    mat = id.gaborDest;
}

/*
Perform the Canny edge detector trials.
- Pavel Shekhter
*/
void cannyTrial (std::ofstream &file, char ** argv, int i, int trial, std::ofstream &csv, int argc) {
    cv::Mat gaussCannyDet;
    bool isGCDone = false;
    std::thread gaussCanny (&gaussianCanny, std::ref (file), argv[i], std::ref (gaussCannyDet), std::ref (csv));
    if (gaussCanny.joinable ()) {
        gaussCanny.join ();
        isGCDone = true;
    }

    if (!gaussCannyDet.empty () || isGCDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_gaussian_" + imp, gaussCannyDet, comp_params);
                cv::Mat gaussianCannyInv;
                cv::bitwise_not (gaussCannyDet, gaussianCannyInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_gaussian_inv_" + imp, gaussianCannyInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!gaussCannyDet.empty ()) {
        cv::namedWindow ("Canny: Gaussian", CV_WINDOW_NORMAL);
        cv::imshow ("Canny: Gaussian", gaussCannyDet);
        cv::namedWindow ("Canny: Gaussian Blur Inverted", CV_WINDOW_NORMAL);
        cv::Mat gaussCannyInv;
        cv::bitwise_not (gaussCannyDet, gaussCannyInv);
        cv::imshow ("Canny: Gaussian Blur Inverted", gaussCannyInv);
    }

    cv::Mat normalizedCannyDet;
    bool isNCDone = false;
    std::thread normCanny (&normalizedCanny, std::ref (file), argv[i], std::ref (normalizedCannyDet), std::ref (csv));
    if (normCanny.joinable ()) {
        normCanny.join ();
        isNCDone = true;
    }

    if (!normalizedCannyDet.empty () || isNCDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_normalized_box_" + imp, normalizedCannyDet, comp_params);
                cv::Mat normCannyInv;
                cv::bitwise_not (normalizedCannyDet, normCannyInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_normalized_inv_" + imp, normCannyInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!normalizedCannyDet.empty ()) {
        cv::namedWindow ("Canny: Normalized Box", CV_WINDOW_NORMAL);
        cv::imshow ("Canny: Normalized Box", gaussCannyDet);
        cv::namedWindow ("Canny: Normalized Box Inverted", CV_WINDOW_NORMAL);
        cv::Mat normalizeddCannyInv;
        cv::bitwise_not (normalizedCannyDet, normalizeddCannyInv);
        cv::imshow ("Canny: Normalized Box Inverted", normalizeddCannyInv);
    }


    cv::Mat boxCannyDet;
    bool isBoxDone = false;
    std::thread boxCanny (&boxCanny, std::ref (file), argv[i], std::ref (boxCannyDet), std::ref (csv));
    if (boxCanny.joinable ()) {
        boxCanny.join ();
        isBoxDone = true;
    }

    if (!boxCannyDet.empty () || isBoxDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_box_" + imp, boxCannyDet, comp_params);
                cv::Mat boxCannyInv;
                cv::bitwise_not (boxCannyDet, boxCannyInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_canny_box_inv_" + imp, boxCannyInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!boxCannyDet.empty ()) {
        cv::namedWindow ("Canny: Box Filter", CV_WINDOW_NORMAL);
        cv::imshow ("Canny: Box Filter", boxCannyDet);
        cv::namedWindow ("Canny: Box Filter Inverted", CV_WINDOW_NORMAL);
        cv::Mat boxCannyInv;
        cv::bitwise_not (boxCannyDet, boxCannyInv);
        cv::imshow ("Canny: Box Filter Inverted", boxCannyInv);
    }
}

/*
Perform the Laplacian edge detector trials.
- Pavel Shekhter
*/
void laplaceTrial (std::ofstream &file, char ** argv, int i, int trial, std::ofstream &csv, int argc) {

    cv::Mat gaussLaplaceDet;
    bool isGLDone = false;
    std::thread gaussLaplace (&gausianLaplace, std::ref (file), argv[i], std::ref (gaussLaplaceDet), std::ref (csv));
    if (gaussLaplace.joinable ()) {
        gaussLaplace.join ();
        isGLDone = true;
    }

    if (!gaussLaplaceDet.empty () || isGLDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_gaussian_" + imp, gaussLaplaceDet, comp_params);
                cv::Mat gaussLaplaceInv;
                cv::bitwise_not (gaussLaplaceDet, gaussLaplaceInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_gaussian_inv_" + imp, gaussLaplaceInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!gaussLaplaceDet.empty ()) {
        cv::namedWindow ("Laplacian: Gaussian", CV_WINDOW_NORMAL);
        cv::imshow ("Laplacian: Gaussian", gaussLaplaceDet);
        cv::namedWindow ("Laplacian: Gaussian Blur Inverted", CV_WINDOW_NORMAL);
        cv::Mat gaussLaplaceInv;
        cv::bitwise_not (gaussLaplaceDet, gaussLaplaceInv);
        cv::imshow ("Laplacian: Gaussian Blur Inverted", gaussLaplaceInv);
    }

    cv::Mat normalizedLaplaceDet;
    bool isNLDone = false;
    std::thread normLaplace (&normalizedLaplace, std::ref (file), argv[i], std::ref (normalizedLaplaceDet), std::ref (csv));
    if (normLaplace.joinable ()) {
        normLaplace.join ();
        isNLDone = true;
    }

    if (!normalizedLaplaceDet.empty () || isNLDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_normalized_" + imp, normalizedLaplaceDet, comp_params);
                cv::Mat normLaplaceInv;
                cv::bitwise_not (normalizedLaplaceDet, normLaplaceInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_normalized_inv_" + imp, normLaplaceInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!normalizedLaplaceDet.empty ()) {
        cv::namedWindow ("Laplacian: Normalized", CV_WINDOW_NORMAL);
        cv::imshow ("Laplacian: Normalized", normalizedLaplaceDet);
        cv::namedWindow ("Laplacian: Normalized Inverted", CV_WINDOW_NORMAL);
        cv::Mat normLaplaceInv;
        cv::bitwise_not (normalizedLaplaceDet, normLaplaceInv);
        cv::imshow ("Laplacian: Normalized Inverted", normLaplaceInv);
    }

    cv::Mat boxLaplaceDet;
    bool isBLDone = false;
    std::thread boxLaplace (&boxLaplace, std::ref (file), argv[i], std::ref (boxLaplaceDet), std::ref (csv));
    if (boxLaplace.joinable ()) {
        boxLaplace.join ();
        isBLDone = true;
    }

    if (!boxLaplaceDet.empty () || isBLDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_box_" + imp, boxLaplaceDet, comp_params);
                cv::Mat boxLaplaceInv;
                cv::bitwise_not (boxLaplaceDet, boxLaplaceInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_box_inv_" + imp, boxLaplaceInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!boxLaplaceDet.empty ()) {
        cv::namedWindow ("Laplacian: Box Filter", CV_WINDOW_NORMAL);
        cv::namedWindow ("Laplacian: Box Filter Inverted", CV_WINDOW_NORMAL);
        cv::imshow ("Laplacian: Box Filter", boxLaplaceDet);
        cv::Mat boxLaplaceInv;
        cv::bitwise_not (boxLaplaceDet, boxLaplaceInv);
        cv::imshow ("Laplacian: Box Filter Inverted", boxLaplaceInv);
    }
}

/*
Perform the Sobel edge detector trials.
- Pavel Shekhter
*/
void sobelTrial (std::ofstream &file, char ** argv, int i, int trial, std::ofstream &csv, int argc) {

    cv::Mat gaussSobelMat;
    bool isGSDone = false;
    std::thread gaussSobel (&gaussianSobel, std::ref (file), argv[i], std::ref (gaussSobelMat), std::ref (csv));
    if (gaussSobel.joinable ()) {
        gaussSobel.join ();
        isGSDone = true;
    }

    if (!gaussSobelMat.empty () || isGSDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_sobel_gaussian_" + imp, gaussSobelMat, comp_params);
                cv::Mat gaussSobelInv;
                cv::bitwise_not (gaussSobelMat, gaussSobelInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_sobel_gaussian_inv_" + imp, gaussSobelInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!gaussSobelMat.empty ()) {
        cv::namedWindow ("Sobel: Gaussian", CV_WINDOW_NORMAL);
        cv::imshow ("Sobel: Gaussian", gaussSobelMat);
        cv::namedWindow ("Sobel: Gaussian Blur Inverted", CV_WINDOW_NORMAL);
        cv::Mat gaussSobelInv;
        cv::bitwise_not (gaussSobelMat, gaussSobelInv);
        cv::imshow ("Sobel: Gaussian Blur Inverted", gaussSobelInv);
    }

    cv::Mat normalizedSobelMat;
    bool isNSDone = false;
    std::thread normSobel (&normalizedSobel, std::ref (file), argv[i], std::ref (normalizedSobelMat), std::ref (csv));
    if (normSobel.joinable ()) {
        normSobel.join ();
        isNSDone = true;
    }

    if (!normalizedSobelMat.empty () || isNSDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_sobel_normalized_" + imp, normalizedSobelMat, comp_params);
                cv::Mat normSobelInv;
                cv::bitwise_not (normalizedSobelMat, normSobelInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_sobel_normalized_inv_" + imp, normSobelInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!normalizedSobelMat.empty ()) {
        cv::namedWindow ("Sobel: Normalized", CV_WINDOW_NORMAL);
        cv::imshow ("Sobel: Normalized", normalizedSobelMat);
        cv::namedWindow ("Sobel: Normalized Inverted", CV_WINDOW_NORMAL);
        cv::Mat normSobelInv;
        cv::bitwise_not (normalizedSobelMat, normSobelInv);
        cv::imshow ("Sobel: Normalized Inverted", normSobelInv);
    }

    cv::Mat boxSobelMat;
    bool isBSDone = false;
    std::thread boxSobel (&boxSobel, std::ref (file), argv[i], std::ref (boxSobelMat), std::ref (csv));
    if (boxSobel.joinable ()) {
        boxSobel.join ();
        isBSDone = true;
    }

    if (!boxSobelMat.empty () || isBSDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_sobel_box_" + imp, boxSobelMat, comp_params);
                cv::Mat boxSobelInv;
                cv::bitwise_not (boxSobelMat, boxSobelInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_laplace_box_inv_" + imp, boxSobelInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    if (!boxSobelMat.empty ()) {
        cv::namedWindow ("Sobel: Box Filter", CV_WINDOW_NORMAL);
        cv::namedWindow ("Sobel: Box Filter Inverted", CV_WINDOW_NORMAL);
        cv::imshow ("Sobel: Box Filter", boxSobelMat);
        cv::Mat boxSobelInv;
        cv::bitwise_not (boxSobelMat, boxSobelInv);
        cv::imshow ("Sobel: Box Filter Inverted", boxSobelInv);
    }
}

/*
Perform the Gabor edge detector trials.
- Pavel Shekhter
*/
void gaborTrial (std::ofstream &file, char ** argv, int i, int trial, std::ofstream &csv, int argc) {
    cv::Mat gaborDet;
    bool isGDone = false;
    std::thread gabor (&gabor, std::ref (file), argv[i], std::ref (gaborDet), std::ref (csv));
    if (gabor.joinable ()) {
        gabor.join ();
        isGDone = true;
    }

    if (!gaborDet.empty () || isGDone == true) {
        std::vector<int> comp_params;
        comp_params.push_back (CV_IMWRITE_JPEG_QUALITY);
        comp_params.push_back (100);
        std::string im = argv[i];
        boost::filesystem::path image_path (im);
        if (boost::filesystem::exists (image_path)) {
            std::string imp = image_path.filename ().generic_string ();
            try {
                bool save = cv::imwrite ("trial_" + std::to_string (trial) + "_gabor_" + imp, gaborDet, comp_params);
                cv::Mat gaborInv;
                cv::bitwise_not (gaborDet, gaborInv);
                save = cv::imwrite ("trial_" + std::to_string (trial) + "_gabor_inv_" + imp, gaborInv, comp_params);
            } catch (std::runtime_error& e) {
                appendErrorMessage (file, -3);
                fprintf (stderr, "Unable to write file due to: %s\n", e.what ());
            }
        }
    }

    
    if (!gaborDet.empty ()) {
        cv::namedWindow ("Gabor", CV_WINDOW_NORMAL);
        cv::imshow ("Gabor", gaborDet);
        cv::namedWindow ("Gabor Inverted", CV_WINDOW_NORMAL);
        cv::Mat gaborInv;
        cv::bitwise_not (gaborDet, gaborInv);
        cv::imshow ("Gabor Inverted", gaborInv);
    }

}

int main (int argc, char* argv[]) {
    int cycles = 0;
    std::string cyclesString;
    std::ofstream file;
    std::ofstream csv;

    if (!(argc > 1)) {
        std::cout << "Usage: CompVisionProject imageToLoad" << std::endl;
        appendErrorMessage (std::cout, -2);
        return -2;
    }

    std::string report;

    std::cout << "Enter report file name: " << std::endl;
    getline (std::cin, report);

    std::string csvString;

    std::cout << "Enter name of CSV: " << std::endl;
    getline (std::cin, csvString);

    std::cout << "How many trials do you want?" << std::endl;
    getline (std::cin, cyclesString);
    cycles = std::stoi (cyclesString);

    file.open (report);
    csv.open (csvString);

    setUpFile (file, report, csv);

    for (int trials = 0; trials < cycles; ++trials) {
        file << "Starting trial " << trials << std::endl;
        for (int i = 1; i < argc; i++) {
            bool retflag;
            int retval = parseArguments (argc, argv[i], file, retflag);
            if (retflag) return retval;

            cv::namedWindow ("Computer Vision Demo", CV_WINDOW_NORMAL);
            cv::imshow ("Computer Vision Demo", id.currentFrameColor);
            cv::cvtColor (id.currentFrameColor, id.currentFrameGry, cv::COLOR_BGR2GRAY);

            for (int currentArg = 1; currentArg < argc; ++currentArg) {
                csv << "Trial #" << i << " File #" << currentArg << ", ";
                laplaceTrial (file, argv, i, trials, csv, currentArg);
                cannyTrial (file, argv, i, trials, csv, currentArg);
                sobelTrial (file, argv, i, trials, csv, currentArg);
                gaborTrial (file, argv, i, trials, csv, currentArg);
                csv << "\n";
            }

            printf ("Finished running edge detection Trial #%d. Press Esc to quit. Images and report will be found in folder where CompVisionDemo.exe is located.\n", i);
            cv::waitKey (0);

        }
        csv << "\n";
        file << "Trial #" << trials << " ended.\n" << std::endl;
    }

    file.close ();

    return 0;
}
