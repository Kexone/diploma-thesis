#include "videostream.h"

VideoStream::VideoStream(int cam)
{
    this->camera = cam;
    capture.open(cam);
}

cv::Mat VideoStream::getFrame()
{
    cv::Mat cameraFrame;
    if (!capture.isOpened()) {
            return cameraFrame.empty();
    }
    capture >> cameraFrame;
    return cameraFrame;

}
