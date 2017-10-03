#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::chooseType(int type, std::vector<cv::Mat> frames)
{
    cv::Mat test;
    mog = Mog();
    hog = Hog();
    cv::Mat frame;
    cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);
    for(auto &frame: frames) {
   // for(uint i= 0; i < frames.size(); i++) {
     //   frame = frames[i];
       // if(frame.empty()) {
         //   cv::destroyWindow("Test");
           // return;
        //}
        //test = mog.processMat(frame);
        cv::imshow("Test", mog.processMat(frame));
        cv::waitKey(20);
       // frame.release();
    }
    cv::destroyWindow("Test");
}
