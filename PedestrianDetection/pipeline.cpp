#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::chooseType(int type, std::vector<cv::Mat> frames)
{
    cv::Mat test;
    mog = Mog();
    test = mog.processMat(frames[0]);
    cv::imshow("Test", test);
}
