#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>
#include <type_traits>

#include "../train/trainfhog.h"

/*
 * class Utils
 */
class Utils
{
public:

	/**
	* @brief This function creates new directory if not exists. Using the bad practise (call system() func), use carefully
	*
	* @param path path with name making directory
	*/
	static void makeDir(std::string path)
	{
#pragma GCC diagnostic ignored "-Wwrite-strings"
		char * mkDirStr = static_cast<char *>("mkdir ");
		int lenComm = 6;
		char *temp = new char[path.length()];
		std::strcpy(temp, mkDirStr);
		for (int i = 0; i < path.length(); i++)
			temp[lenComm + i] = path[i];
		struct stat st;
		if (stat(temp, &st) != 0) {
			system(temp);
		}
	}

	/**
	* @brief Converts dlib list of rectangles to openCV rectangles
	*
	* @param r list of dlib rectangles
	*/
	static std::vector < cv::Rect > vecDlibRectangle2VecOpenCV(std::vector< dlib::rectangle > r)
	{
		std::vector < cv::Rect > rects;
		for (int i = 0; i < r.size(); i++)
			rects.push_back(cv::Rect(cv::Point2i(r[i].left(), r[i].top()), cv::Point2i(r[i].right() + 1, r[i].bottom() + 1)));

		return rects;
	}

	/**
	* @brief Fills vector of cv::Mat from string path and sets the labels.
	* @sNeg is switcher for negative and positive samples which determines what will be filled to list of labels
	*
	* @param path path to samples
	* @param dstList list of cv::Mat
	* @param labels list of labels
	* @param  pedSize size of pedestrian (image)
	* @param isNeg switcher between samples
	* @param forDlib switcher between OpenCV and DLib causes switch types of labels
	*/
	static void fillSamples2List(std::string &path, std::vector<cv::Mat> &dstList, std::vector< int > &labels, cv::Size pedSize, bool isNeg = false, bool forDlib = false)
	{
		assert(!path.empty());
		cv::Mat frame;
		std::fstream sampleFile(path);
		std::string oSample;
		while (sampleFile >> oSample) {
			//	std::cout << oSample << std::endl;
			frame = cv::imread(oSample, CV_32FC3);
			if (frame.empty()) {
				std::cout << "fail " << oSample << std::endl;
				continue;
			}
			cv::resize(frame, frame, pedSize);
			dstList.push_back(frame.clone());
			if (!forDlib) {
				if (!isNeg) {
					labels.push_back(1);
				}
				else {
					labels.push_back(0);
				}
			}
			else
			{
				if (!isNeg) {
					labels.push_back(+1);
				}
				else {
					labels.push_back(-1);
				}
			}
		}
	}

	/**
	* @brief Fills vector ofdlib::matrix < TrainFHog::pixel_type> from string path and sets the labels.
	* @sNeg is switcher for negative and positive samples which determines what will be filled to list of labels
	* @ TODO remove ? 
	* @param path path to samples
	* @param dstList list of dlib::matrix < TrainFHog::pixel_type>
	* @param labels list of labels
	* @param  pedSize size of pedestrian (image)
	* @param isNeg switcher between samples
	*/
	static void fillSamples2List(std::string &path, std::vector< dlib::matrix < TrainFHog::pixel_type> > &dstList, std::vector<float> &labels, cv::Size pedSize, bool isNeg = false)
	{
		//	int i = 0;
		assert(!path.empty());
		cv::Mat frame;
		//dlib::array2d < dlib::bgr_pixel > frame;
		std::fstream sampleFile(path);
		std::string oSample;
		while (sampleFile >> oSample) {
			frame = cv::imread(oSample, CV_32FC3);
			if (frame.empty())		std::cout << "fail" << std::endl;
			cv::resize(frame, frame, pedSize);
			dlib::cv_image<TrainFHog::pixel_type> cvTmp(frame);
			dlib::matrix<TrainFHog::pixel_type> test = dlib::mat(cvTmp);
			dstList.push_back(test);
		}
		if (!isNeg) {
			labels.push_back(1);
		}
		else {
			labels.push_back(-1);
		}
	}

	/**
	 * @brief This method parsing image by sliding window to samples to train
	 *
	 * @param path path to samples
	 * @param folder folder where save parsed images
	 */
	static void createSamplesFromImage(std::string path, std::string folder)
	{
		assert(!path.empty());
		assert(!folder.empty());
		cv::Mat img;
		std::fstream sampleFile;
		std::string oSample;
		int nSample = 0;
		char imgName[30];
		sampleFile.open(path);

		makeDir(folder);

		folder.append("/");
		while (sampleFile >> oSample) {
			img = imread(oSample, cv::IMREAD_COLOR);
			cv::Mat bcp = img.clone();
			int xStep = 48, yStep = 96, nCount = 0;
			for (int y = 0; y < img.rows - yStep; y += yStep)
			{
				for (int x = 0; x < img.cols - xStep; x += xStep)
				{
					cv::Rect rect(cv::Point(x, y), cv::Point(x + xStep, y + yStep));
					cv::Mat roi = img(rect);
					std::sprintf(imgName, "negsample%d_%d.jpg", nSample, nCount);
					cv::imwrite(folder + imgName, roi);
					//cv::rectangle(img, rect, cv::Scalar(0, 255, 0));
					//cv::imshow("test", img);
					//cv::waitKey(1);
					//img = bcp.clone();
					roi.release();
					nCount++;
				}
			}
			nSample++;
		}
	}
};

#endif // UTILS_H