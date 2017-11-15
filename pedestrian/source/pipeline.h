#ifndef PIPELINE_H
#define PIPELINE_H
#include <iostream>
#include <sstream>

#include "alg/mog.h"
#include "alg/hog.h"
#include "alg/convexhull.h"
#include "media/croppedimage.h"
#include "settings.h"
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
    Hog hog;
	CascadeClass cc;
    ConvexHull ch;
    VideoStream *vs;
    cv::Mat localFrame;
    std::vector<std::vector<cv::Rect>> found_filtered;

    void process(cv::Mat frame);
	void preprocessing(cv::Mat &frame);
    void draw2mat(std::vector<CroppedImage> croppedImages);


};

#endif // PIPELINE_H
