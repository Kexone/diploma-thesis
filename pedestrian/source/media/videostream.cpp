#include "videostream.h"

int VideoStream::fps = 0;
int VideoStream::totalFrames = 0;
cv::Size VideoStream::vidRes = cv::Size(0, 0);
VideoStream::VideoStream()
{
}

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
        return cv::Mat();
    capture >> cameraFrame;
    return cameraFrame;
}

void VideoStream::openCamera()
{
    if(camera !=99)
        capture.open(camera);
    else  {
        capture.open(camSource);
		VideoStream::totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        VideoStream::fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
		VideoStream::vidRes = cv::Size(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    }
}

