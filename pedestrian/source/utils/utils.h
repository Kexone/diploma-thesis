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
	* @brief This function sets the names for GT and test file
	*
	* @param videoName path to video file
	*/
	static void setEvaluationFiles(std::string videoName)
	{
		Settings::nameFile = videoName;
		replace(Settings::nameFile.begin(), Settings::nameFile.end(), '/', '-');
		replace(Settings::nameFile.begin(), Settings::nameFile.end(), '.', '-');
		Settings::nameTrainedFile = "data//trained//" + Settings::nameFile;
		Settings::nameFile = "data//tested//" + Settings::nameFile;
		Settings::nameTrainedFile.append("_trained.txt");
		Settings::nameFile.append(".txt");
	}

	/**
	* @brief Converts dlib list of rectangles to openCV rectangles
	*
	* @param r list of dlib rectangles
	* @param minRectSize param to filter minimal size of rect (default is 0)
	*/
	static std::vector < cv::Rect > vecDlibRectangle2VecOpenCV(std::vector< dlib::rectangle > r, int minRectSize = 0)
	{
		std::vector < cv::Rect > rects;
		for (int i = 0; i < r.size(); i++) {
			cv::Rect rect = cv::Rect(cv::Point2i(r[i].left(), r[i].top()), cv::Point2i(r[i].right() + 1, r[i].bottom() + 1));
			if(rect.area() > minRectSize)
				rects.push_back(rect);
		}
		return rects;
	}

	/**
	* @brief Fills vector of cv::Mat from string path and sets the labels.
	* @sNeg is switcher for negative and positive samples which determines what will be filled to list of labels
	*
	* @param path path to samples
	* @param dstList list of cv::Mat
	* @param  pedSize size of pedestrian (image)
	*/
	static void fillSamples2List(std::string &path, std::vector<cv::Mat> &dstList, cv::Size pedSize)
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
		}
	}

	/**
	 * @brief This method crops the imags in sliding window to samples for train (size is defined in external settings file)
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
			int xStep = 64, yStep = 128, nCount = 0;
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