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
	*
	*/
	void evaluate();
	static int allDetections;

private:
    Mog _mog;
	Hog		_hog;
//	FHog _fhog;
	//Hog _hog = Hog("3.yml");
	//Hog _hog = Hog("48_96_16_8_8_9_01.yml");


	//	Hog hog = Hog("2292_78_98.3.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");

	
	//CascadeClass _cc;
    ConvexHull _ch;
    VideoStream *_vs;
    cv::Mat _localFrame;
	std::vector < std::vector < std::vector < cv::Rect > > > _rects2Eval;

	int _dilation_type = cv::MORPH_CROSS;
	int _erosion_type = cv::MORPH_CROSS;
	int _dilation_size = 6;
	int _erosion_size = 1;
	int _typeAlgorithm;

	cv::Mat _dilMat, _eroMat;
	/**
	* @brief
	* @TODO prepare to delete
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
    void process(cv::Mat &frame, int cFrame);

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
	void mixturedHoG(cv::Mat &frame, int cFrame);

	/**
	* @brief This testing function uses Gaussian mixture to analyzes and substraction of motion segments and thereafter uses HoG from dlib
	*
	* @param frame actual frame
	* @param cFrame count frame for saving location of detection
	*/
	void mixturedFHoG(cv::Mat &frame, int cFrame);

	/**
	* @brief @TODO this doc
	*
	* @param frame actual frame
	*/
	void processStandaloneImage(cv::Mat &frame);

	/**
	* @brief @TODO this doc
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
	* @brief @TODO this doc
	*
	* @param croppedImages
	* @param rect
	*/
    void draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect);

	/**
	* @brief @TODO this doc
	*
	* @param rect
	*/
	void draw2mat(std::vector < cv::Rect > &rect);

	/**
	* @brief  @TODO this doc
	*
	* @param filePath
	*/
	void saveResults();

	/**
	* @brief @TODO this doc
	*
	* @param filePath
	* @param rects
	*/
	void loadRects(std::string filePath, std::vector< std::vector<cv::Rect> > & rects);

	/**
	* @brief @TODO this doc
	*
	* @param rects
	* @param croppedImages
	* @param rects2Save
	*/
	void rectOffset(std::vector<std::vector<cv::Rect>> &rects, std::vector< CroppedImage > &croppedImages, std::vector<std::vector<cv::Rect>> &rects2Save);

};

#endif // PIPELINE_H
