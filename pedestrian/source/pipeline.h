#ifndef PIPELINE_H
#define PIPELINE_H

#include <iostream>
#include <sstream>

#include "alg/mog.h"
#include "alg/hog.h"
#include "alg/fhog.h"
#include "alg/convexhull.h"
#include "media/croppedimage.h"
#include "media/videostream.h"
#include "alg/cascadeClass.h"

/**
 * class Pipeline
 */
class Pipeline
{
public:
	Pipeline() : Pipeline("default") {};
	Pipeline(std::string svmPath);

	/**
	* @brief Executed for images. Function for ran detection on images. 
	*
	* @param testSamplesPath path to file with images paths
	*/
    void executeImages(std::string testSamplesPath);

	/**
	* @brief Executed for camera feed. Function for ran in videostream
	*
	* @param cameraFeed choose camera type
	*/
    void execute(int cameraFeed);

	/**
	* @brief Executed for videostream. Function for ran detection in videostream
	*
	* @param cameraFeed path to video file
	* @param algorithmType type of executed algorithm to test it (e.g. with or without mixture Gaussian)
	*/
    void execute(std::string cameraFeed, int algorithmType);

	/**
	* @brief Evalution function. Compares the position of rects with trained position of pedestrian in frame. It passes line by line for all frames.
	*/
	void evaluate();

	static int allDetections;

private:
    Mog _mog;
	Hog _hog;
	FHog _fhog;
	CascadeClass _cc;
    ConvexHull _ch;
    VideoStream *_vs;

	cv::Mat _localFrame;
	std::vector < std::vector < std::vector < cv::Rect > > > _rects2Eval;

	int _dilation_type = cv::MORPH_CROSS;
	int _erosion_type = cv::MORPH_CROSS;
	int _dilation_size = 3;
	int _erosion_size = 2;
	int _typeAlgorithm;

	cv::Mat _dilMat, _eroMat;
	
	/**
	* @brief This testing function uses only HoG from OpenCV on full image.
	*
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
	void pureHoG(cv::Mat &frame, int cFrame);

	/**
	* @brief This testing function uses only FHoG from dlib on full image. Like published by Dalal and Triggs in 2005 in the paper
    Histograms of Oriented Gradients for Human Detection.
	*
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
	void pureFHoG(cv::Mat &frame, int cFrame);

	/**
	* @brief This testing function uses Gaussian mixture to analyzes and substraction of motion segments and thereafter uses HoG from openCV
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
	void mogAndHog(cv::Mat &frame, int cFrame);

	/**
	* @brief This testing function uses Gaussian mixture to analyzes and substraction of motion segments and thereafter uses HoG from dlib
	*
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
	void mogAndFHog(cv::Mat &frame, int cFrame);

	/**
	* @brief Method for processing one image where used classic HoG to detection pedestrian
	*
	* @param frame actual frame
	*/
	void processStandaloneImage(cv::Mat &frame);

	/**
	* @brief Method for preprocessing, convert to gray is standard
	*
	* @param frame actual frame
	*/
	void preprocessing(cv::Mat &frame);

	/**
	* @brief performs dilation and erosion on frame
	*
	* @param frame actual frame
	*/
	void dilateErode(cv::Mat &frame);

	/** 
	* @brief Draw rectangles into mat and also increment the allDetections counter
	*  Used for cropped images to set the default position in frame
	*
	* @param croppedImages
	* @param rect
	*/
    void draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect);

	/**
	* @brief Draw rectangles into mat and also increment the allDetections counter
	* Used for non-cropped images, only drews the rectangles into the frame
	*
	* @param rect
	*/
	void draw2mat(std::vector < cv::Rect > &rect);

	/**
	* @brief Saving method. This method saves the rectangles and numbers of frame where was found. On each line is frame number and points of rectangle.
	* The first line is count of all frames
	*/
	void saveResults();

	/**
	* @brief Loads the rectangles from file for evaluate detection
	*
	* @param filePath
	* @param rects
	*/
	void loadRects(std::string filePath, std::vector< std::vector<cv::Rect> > & rects);

	/**
	* @brief This method adds the offset of cropped images to be have default position in frame
	*
	* @param rects
	* @param croppedImages
	* @param rects2Save
	*/
	void rectOffset(std::vector<std::vector<cv::Rect>> &rects, std::vector< CroppedImage > &croppedImages, std::vector<std::vector<cv::Rect>> &rects2Save);

};

#endif // PIPELINE_H
