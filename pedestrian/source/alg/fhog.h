#ifndef FHOG_H
#define FHOG_H

#include <dlib/svm_threaded.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/core/core_c.h>
#include "../media/croppedimage.h"
#include <iostream>
#include <fstream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>

/**
 * class FHog
 */
class FHog
{
public:
	FHog();
//	FHog(int def);
	//FHog(std::string svmPath);

	/**
	* @brief
	*
	* @param
	*/
	std::vector< std::vector < cv::Rect  > > detect(std::vector< CroppedImage > &frame);
	//void detect(std::vector< CroppedImage > &frame);

	/**
	* @brief
	*
	* @param
	*/
	void detect(const std::vector< cv::Mat > testLst, int &nTrue, int &nFalse, bool pedestrian = true);

private:
	//void getSvmDetector(const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector);
	//cv::HOGDescriptor hog;
	//cv::Ptr<cv::ml::SVM> svm;
};

#endif // FHOG_H
