#pragma once
#include <opencv2/core/matx.hpp>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <dlib/pixel.h>
#include <dlib/opencv/cv_image.h>

class DlibSvmTest
{
public:
	DlibSvmTest(std::vector<cv::Mat> samplesList, std::vector<float> labels) : samplesList(samplesList), fLabels(labels) {}
	cv::Vec4f process();
private:
	std::vector<cv::Mat> samplesList;
	std::vector<float> fLabels;
	using pixel_type = dlib::bgr_pixel;

	void writeResult2File(cv::Vec4f resultVec);
};
