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
/**
 * class MyFHog
 */
class MyFHog
{
public:
	MyFHog();

	/**
	* @brief Constructor of my FHoG class which sets the SVM
	*
	* @param svmPath is path to svm, if sent word is 'default' will sets the attached svm
	*/
	MyFHog(std::string classPath);

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
	//typedef dlib::matrix < double, 1980, 1  >  image_scanner_type;
//	typedef dlib::matrix < double, 1980, 1 > image_type;
	//typedef dlib::radial_basis_kernel< image_type  > image_scanner_type;
 	//typedef  dlib::radial_basis_kernel < dlib::matrix < double, 1980, 1 >> image_scanner_type;
		//typedef dlib::radial_basis_kernel  image_scanner_type;
	typedef dlib::scan_fhog_pyramid<dlib::pyramid_down < 3 >, dlib::default_fhog_feature_extractor> image_scanner_type;
	//typedef dlib::structural_svm_object_detection_problem<double, dlib::default_fhog_feature_extractor> image_scanner_type;
//	dlib::scan_fhog_pyramid<dlib::radial_basis_kernel< dlib::matrix < double, 1980, 1 >>> detector;
//	image_scanner_type scanner;
	dlib::object_detector<image_scanner_type> detector;
	//dlib::scan_fhog_pyramid<image_scanner_type> detector;
};

#endif // FHOG_H
