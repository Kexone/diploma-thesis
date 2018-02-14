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
	* @brief Constructor of my HoG class which sets the SVM
	*
	* @param svmPath is path to svm, if sent word is 'default' will sets the default openCV  people detector
	*/
	Hog(std::string svmPath);

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

private:

	/**
	* @brief Gets SVM detector which is needed to detection
	*
	* @param svm svm pointer
	* @param vector of floats
	*/
    void getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector );

	/**
	* @brief This function calls function detectMultiScale from openCV
	*
	* @param img image to processing
	* @param found vector of found rectangles
	*/
	void hogDetectMultiScale(cv::Mat img, std::vector< cv::Rect > &found);
	
	/**
	* @brief Gets value of predict, firstly calculates HoG features and then return predicate.
	* 
	*
	* @param img image to processing
	* @param flags flags for svm predict function
	*/
	float predict(cv::Mat img, int flags = 0);

	/**
	* @brief This method calc the distance of detection
	* 
	* @param img image to processing 
	*/
	float getDistance(cv::Mat img);

    cv::HOGDescriptor hog;
	cv::Ptr<cv::ml::SVM> svm;
};

#endif // HOG_H
