#ifndef TRAINFHOG_H
#define TRAINFHOG_H
#include <iostream>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <dlib/pixel.h>
#include <dlib/opencv/cv_image.h>

class TrainFHog
{
public: 
	using pixel_type = dlib::bgr_pixel;

	TrainFHog() {
		_nu_par = 0001;
		_gamma_par = 0001;
	};
	TrainFHog(double nu, double gamma) : _nu_par(nu), _gamma_par(gamma) { };
	TrainFHog(double nu, double gamma, std::string namefile) : _nu_par(nu), _gamma_par(gamma), _namefile(namefile)  { };

	/**
	* @brief train for dlib SVM
	*
	* @param posSamples path to positive samples
	* @param negSamples path to negative samples
	*/
	void train(std::string posSamples, std::string negSamples);

protected:
	/**
	* @brief Combined train, features are calculated via openCV HoG and after used for train dlib SVM
	*
	* @param gradientSamplesList list of all gradient samples
	* @param labels labels for all samples
	*/
	void train(std::vector< cv::Mat > gradientSamplesList, std::vector< int > labels);
private:

	double _nu_par;
	double _gamma_par;
	std::string _namefile = "pedestrian.svm";

	/**
	* @brief Checks whether the current dataset has any boxes
	*
	* @param boxes list of boxes
	*/
	bool containsAnyBoxes(const std::vector<std::vector<dlib::rectangle> >& boxes);

	/**
	* @brief Finds the average aspect ratio of the elements of boxes and outputs a width
	* and height such that the aspect ratio is equal to the average and also the
	* area is equal to target_size.  That is, the following will be approximately true:
	* #width*#height == target_size
	* #width/#height == the average aspect ratio of the elements of boxes.
	*
	* @param boxes list of boxes to picking
	* @param width set the best width which found
	* @param height set the best height which found
	* @param target_size target size of window
	*/
	void pickBestWindowSize(const std::vector<std::vector<dlib::rectangle> >& boxes, unsigned long& width, unsigned long& height, const unsigned long target_size);

	/**
	* @brief Throws error if boxes didn't have similar size or were smaller than @target_size
	*
	* @param dataset_filename dataset name
	* @param removed list of removed boxes
	* @param target_size target size of window
	*/
	void throwInvalidBoxErrorMessage(const std::string& dataset_filename, const std::vector<std::vector<dlib::rectangle> >& removed, const unsigned long target_size);
};

#endif // TRAINFHOG_H