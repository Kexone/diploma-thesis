#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::chooseType(int type, std::vector<cv::Mat> frames)
{
    cv::Mat test;
    mog = Mog();
    cv::Mat frame;
    cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);
   // for(auto &frame: frames) {
    for(uint i= 0; i < frames.size(); i++) {
        frame = frames[i];
        if(frame.empty())
            return;
        //test = mog.processMat(frame);
        cv::imshow("Test", frame);
        cv::waitKey(525);
       // frame.release();
    }
}
