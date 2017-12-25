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
    Pipeline();

	/**
	* @brief
	*
	* @param
	*/
    void executeImages(std::string testSamplesPath);

	/**
	* @brief
	*
	* @param
	*/
    void execute(int cameraFeed);

	/**
	* @brief
	*
	* @param
	*/
    void execute(std::string cameraFeed);

	/**
	* @brief
	*
	* @param
	*/
	void evaluate(std::string testResultPath, std::string trainedPosPath);
	static int allDetections;

private:
    Mog mog;
//	FHog fhog;
	//Hog hog = Hog("3.yml");
	Hog hog;
	//Hog hog = Hog("48_96_16_8_8_9_01.yml");


	//	Hog hog = Hog("2292_78_98.3.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");

	
	//CascadeClass cc;
    ConvexHull ch;
    VideoStream *vs;
    cv::Mat localFrame;
	std::vector < std::vector < std::vector < cv::Rect > > > rects2Eval;

	/**
	* @brief
	*
	* @param
	*/
    void process(cv::Mat &frame, int cFrame);

	/**
	* @brief
	*
	* @param
	*/
	void processStandaloneIm(cv::Mat &frame);

	/**
	* @brief
	*
	* @param
	*/
	void preprocessing(cv::Mat &frame, bool afterMog = false);

	/**
	* @brief
	*
	* @param
	*/
    void draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect);

	/**
	* @brief
	*
	* @param
	*/
	void draw2mat(std::vector < cv::Rect > &rect);

	/**
	* @brief
	*
	* @param
	*/
	void saveResults(std::string filePath);

	/**
	* @brief
	*
	* @param
	*/
	void loadRects(std::string filePath, std::vector< std::vector<cv::Rect> > & rects);

	/**
	* @brief
	*
	* @param
	*/
	void rectOffset(std::vector<std::vector<cv::Rect>> &rects, std::vector< CroppedImage > &croppedImages, std::vector<std::vector<cv::Rect>> &rects2Save);

};

#endif // PIPELINE_H
