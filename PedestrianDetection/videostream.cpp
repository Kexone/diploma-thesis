#include "videostream.h"
#include "mainwindow.h"
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
    else  {
        capture.open(camSource);
        MainWindow::setTotalFrames(int(capture.get(cv::CAP_PROP_FRAME_COUNT)));
        MainWindow::setFps(int(capture.get(cv::CAP_PROP_FPS)));
    }
}

