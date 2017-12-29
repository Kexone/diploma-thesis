#ifndef HOG_H
#define HOG_H

#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#include <opencv2/highgui.hpp>
#include "../media/croppedimage.h"

/**
 * class Hog
 */
class Hog
{
public:
    Hog();
	/**
	* @brief
	*
	* @param
	*/
	Hog(int def);
	/**
	* @brief
	*
	* @param
	*/
	Hog(std::string svmPath);

	/**
	* @brief
	*
	* @param
	*/
    std::vector< std::vector < cv::Rect > > detect(std::vector< CroppedImage > &frame);

	/**
	* @brief
	*
	* @param
	*/
	std::vector < cv::Rect > Hog::detect(cv::Mat& frame);

	/**
	* @brief
	*
	* @param
	*/
	void detect(const std::vector< cv::Mat > testLst, int &nTrue, int &nFalse, bool pedestrian = true);

private:

	/**
	* @brief
	*
	* @param
	*/
    void getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector );

	float predict(cv::Mat img, int flags = 0);
    cv::HOGDescriptor hog;
	cv::Ptr<cv::ml::SVM> svm;
};

#endif // HOG_H
