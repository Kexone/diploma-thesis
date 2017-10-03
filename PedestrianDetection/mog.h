#ifndef MOG_H
#define MOG_H
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

class Mog
{
public:
    Mog();
    cv::Mat processMat(cv::Mat &frame);
private:
    cv::Ptr<cv::BackgroundSubtractor> pMOG;  //MOG  Background subtractor
    cv::Ptr<cv::BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
};

#endif // MOG_H
