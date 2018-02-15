#ifndef OWNHOG_H
#define OWNHOG_H
#include <vector>
#include <opencv2/core/mat.hpp>
#include "../media/croppedimage.h"

class OwnHog
{
public:
	OwnHog();
	void detectMultiScale(cv::Mat img, std::vector<cv::Rect>& foundLocations,
		double hitThreshold = 0, cv::Size winStride = cv::Size(),
	                      cv::Size padding = cv::Size(), double scale = 1.05,
		double finalThreshold = 2.0, bool useMeanshiftGrouping = false) const;

	bool loadSvm(std::string);

	/**
	* @brief Detection pedestrian on cropped images
	*
	* @param frame the vector of cropped images
	* @param rects vector of vectors rectangles
	*/
	void detect(std::vector< CroppedImage > &frame, std::vector< std::vector < cv::Rect > > &rects, std::vector < std::vector < float > > &distances);

	/**
	* @brief Detection pedestrian on frame
	*
	* @param frame
	* @param rects vector of rectangles
	*/
	void detect(cv::Mat& frame, std::vector < cv::Rect > &rects);

	/**
	* @brief This detection method is for testing, it gets vector of matrix and sets true or false predicate
	*
	* @param testLst list of samples of one type (negative or positive)
	* @param nTrue sets this variable is true
	* @param nFalse sets this variable is false
	* @param pedestrian means on samples is pedestrian
	*/
	void detect(const std::vector< cv::Mat > testLst, int &nTrue, int &nFalse, bool pedestrian = true);
};

#endif //OWNHOG_H