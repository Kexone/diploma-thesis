#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

class VideoStream
{
private:
    std::vector<cv::Mat> videoFrames;
    cv::VideoCapture capture;


public:
    VideoStream();
    std::vector<cv::Mat> getFrames();
    std::string openFile(std::string fileName);

};

#endif // VIDEOSTREAM_H
