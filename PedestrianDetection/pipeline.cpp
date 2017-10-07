#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::chooseType(int type, std::vector<cv::Mat> frames)
{
    //cv::Mat test;x
    mog = Mog();
    hog = Hog();
    std::vector<cv::Rect> found_filtered;
   // cv::Mat frame;
    cv::namedWindow("Test", CV_WINDOW_AUTOSIZE);
    for(auto &frame: frames) {
   // for(uint i= 0; i < frames.size(); i++) {
     //   frame = frames[i];
       // if(frame.empty()) {
         //   cv::destroyWindow("Test");
           // return;
        //}
        //test = mog.processMat(frame);
        found_filtered = hog.detect(mog.processMat(frame));
        for (int i = 0; i < found_filtered.size(); i++)
        {
            cv::Rect r = found_filtered[i];
            const cv::Scalar color(0, 255, 0);
            //cv::Rect recta(0,0,found_filtered[i].width,found_filtered[i].height);
            cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
        }
            cv::imshow("Test", frame);
        cv::waitKey(20);
       // frame.release();
    }
    cv::destroyWindow("Test");
}
