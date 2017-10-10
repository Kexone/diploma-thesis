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
private:
    cv::VideoCapture capture;
    int camera = 99;
    std::string camSource;
public:
    VideoStream(int cam);
    VideoStream(std::string camSource);
    cv::Mat getFrame();
    void openCamera();
    void closeCamera();

};

#endif // VIDEOSTREAM_H
