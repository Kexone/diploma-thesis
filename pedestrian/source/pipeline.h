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
class Pipeline
{
public:
    Pipeline();
    int execute(std::vector<cv::Mat> frames);
    int execute(int cameraFeed);
    int execute(std::string cameraFeed);
private:
    Mog mog;
    Hog hog;
    ConvexHull ch;
    VideoStream *vs;
    cv::Mat localFrame;
    std::vector<std::vector<cv::Rect>> found_filtered;
    std::vector<std::vector<cv::Rect>> rect;
    void process(cv::Mat frame);
    void draw2mat(std::vector<CroppedImage> croppedImages);
    void debugMog(cv::Mat frame);
    void debugCHHOG(cv::Mat frame);
    int allDetections = 0;


};

#endif // PIPELINE_H
