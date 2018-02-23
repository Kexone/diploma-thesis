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
#include <dlib/global_optimization.h>

class DlibSvmTest
{
public:
	DlibSvmTest(cv::Mat trainMat, std::vector<double> labels);
	cv::Vec4f process(int type);
private:
	typedef dlib::matrix < double, 1980, 1 > sample_type;
	typedef dlib::radial_basis_kernel<sample_type> kernel_type;
	std::vector < double > _labels;
	std::vector < sample_type > _samples;

	void writeResult2File(cv::Vec4f resultVec);
	void testCsvm();
	void testNusvm(cv::Vec4f & vec);
};
