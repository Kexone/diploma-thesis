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

class Pipeline
{
public:
    Pipeline();
    void execute(std::vector<cv::Mat> frames);
    void execute(int cameraFeed);
    void execute(std::string cameraFeed);

	static int allDetections;

private:
    Mog mog;
	FHog fhog;
	//Hog hog = Hog("3.yml");
	Hog hog;
	//Hog hog = Hog("48_96_16_8_8_9_01.yml");


	//	Hog hog = Hog("2292_78_98.3.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("2717_78_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");
	//	Hog hog = Hog("3111_79_98.4.yml");

	
	CascadeClass cc;
    ConvexHull ch;
    VideoStream *vs;
    cv::Mat localFrame;

    void process(cv::Mat frame);
	void preprocessing(cv::Mat &frame);
    void draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect);


};

#endif // PIPELINE_H
