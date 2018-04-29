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
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include "../settings.h"

/**
 * @class Fhog
 * 
 * @brief this class is used for detection pedestrian by openCV Hog
 */
class Fhog
{
public:
	Fhog();

	/**
	* @brief Constructor of my FHoG class which sets the SVM
	*
	* @param classPath is path to file
	*/
	Fhog(std::string classPath);

	/**
	* @brief Detection pedestrian on cropped images
	*
	* @param frame the vector of cropped images
	* @param rects vector of vectors rectangles
	*/
	void detect(std::vector<CroppedImage>& frame, std::vector< std::vector < cv::Rect  > > &rects);

	/**
	* @brief Detection pedestrian on frame
	*
	* @param frame
	* @param rects vector of rectangles
	*/
	void detect(cv::Mat& frame, std::vector< cv::Rect > &rects);

	float predict(cv::Mat img, int flags);
private:
	using pixel_type = dlib::bgr_pixel;
	typedef dlib::scan_fhog_pyramid<dlib::pyramid_down < 6 >, dlib::default_fhog_feature_extractor> image_scanner_type;
	dlib::object_detector<image_scanner_type> detector;

};

#endif // FHOG_H
