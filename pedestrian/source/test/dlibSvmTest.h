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

/**
*  class DlibSvmTest
*
*  Testing Dlib SVM classificator
*/
class DlibSvmTest
{
public:
	DlibSvmTest(cv::Mat trainMat, std::vector<double> labels);
	/**
	* @brief Choose between nu svm or c svm to testing
	* 
	* @param type type of svm (1 - nu, other - c)
	*/
	void process(int type);
private:
	typedef dlib::matrix < double, 1980, 1 > sample_type;
	typedef dlib::radial_basis_kernel<sample_type> kernel_type;
	std::vector < double > _labels;
	std::vector < sample_type > _samples;

	void writeResult2File(double *vec, int type);
	/**
	* @brief Testing C SVM
	*
	* @param vec array of results
	*/
	void testCsvm(double *vec);

	/**
	* @brief Testing NU SVM
	*
	* @param vec array of results
	*/
	void testNusvm(double * vec);
};
