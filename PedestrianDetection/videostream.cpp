#include "videostream.h"

VideoStream::VideoStream(int cam)
{
    this->camera = cam;

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
    capture.open(camera);
    if (!capture.isOpened())
        std::cout << "err";
    else
        std::cout << "ok";
}
