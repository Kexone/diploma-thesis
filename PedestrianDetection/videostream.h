#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/videoio.hpp>

class VideoStream
{
//private:
    //std::vector<cv::Mat> videoFrames;
    cv::VideoCapture capture;
    int camera;


public:
    VideoStream(int cam);
    cv::Mat getFrame();
    void openCamera();
    //std::vector<cv::Mat> getFrames();
    //std::string openFile(std::vector<std::string> fileName);

};

#endif // VIDEOSTREAM_H
