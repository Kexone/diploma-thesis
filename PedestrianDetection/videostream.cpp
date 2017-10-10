#include "videostream.h"

VideoStream::VideoStream(int cam)
{
    this->camera = cam;
}
VideoStream::VideoStream(std::string camSource)
{
    this->camSource = camSource;
}

cv::Mat VideoStream::getFrame()
{
    cv::Mat cameraFrame;
    if (!capture.isOpened())
        return cameraFrame;
    capture >> cameraFrame;
    return cameraFrame;
}

void VideoStream::openCamera()
{
    if(camera !=99)
        capture.open(camera);
    else
        capture.open(camSource);
}

void VideoStream::closeCamera()
{
    if(capture.isOpened())
        capture.open(0);
        capture.release();
}
