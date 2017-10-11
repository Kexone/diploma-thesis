#ifndef PIPELINE_H
#define PIPELINE_H
#include <iostream>
#include <sstream>

#include "mog.h"
#include "hog.h"
#include "convexhull.h"
#include "croppedimage.h"
#include "settings.h"
#include "videostream.h"
class Pipeline
{
public:
    Pipeline();
    ~Pipeline();
    void interruptDetection();
    int execute(std::vector<cv::Mat> frames);
    int execute(int cameraFeed);
    int execute(std::string cameraFeed);
private:
    bool interrupt;
    Mog mog;
    Hog hog;
    ConvexHull ch;
    cv::Mat localFrame;
    VideoStream *vs;
    std::vector<std::vector<cv::Rect>> found_filtered;
    std::vector<std::vector<cv::Rect>> rect;
    void process(cv::Mat frame);
    void draw2mat(std::vector<CroppedImage> croppedImages);
    void set();
    void debugMog(cv::Mat frame);
    int allDetections = 0;


};

#endif // PIPELINE_H
