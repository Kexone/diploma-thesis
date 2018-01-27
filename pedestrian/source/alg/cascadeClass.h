#ifndef CASCADECLASS_H
#define CASCADECLASS_H

#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#include <opencv2/highgui.hpp>
#include "../media/croppedimage.h"

/**
 * class CascadeClass
 * 
 * This class represents cascade classificator
 */
class CascadeClass
{
public:
	CascadeClass();

	/**
	* @brief Constructor of my Cascade Classificator class which sets the cascades
	*
	* @param svmPath is path to cascade classificator, if sent word is 'default' will sets the attached classifier
	*/
	CascadeClass(std::string filename);

	/**
	* @brief Detection pedestrian on cropped images
	*
	* @param frame the vector of cropped images
	* @param rects vector of vectors rectangles
	*/
	void detect(std::vector < CroppedImage > &frame, std::vector< std::vector < cv::Rect  > > &rects);

	/**
	* @brief Detection pedestrian on frame
	*
	* @param frame
	* @param rects vector of rectangles
	*/
	void detect(cv::Mat& frame, std::vector< cv::Rect > &rects);


private:
	cv::CascadeClassifier clasifier;
};

#endif //CASCADECLASS_H