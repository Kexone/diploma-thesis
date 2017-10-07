#ifndef PIPELINE_H
#define PIPELINE_H
#include <iostream>
#include <sstream>

#include "mog.h"
#include "hog.h"
#include "convexhull.h"
#include "croppedimage.h"

class Pipeline
{
public:
    Pipeline();
    void chooseType(int type, std::vector<cv::Mat> frames);
private:
    Mog mog;
    Hog hog;
    ConvexHull *ch;


};

#endif // PIPELINE_H
