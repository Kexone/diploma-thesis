#include "videostream.h"
#include "mainwindow.h"
VideoStream::VideoStream()
{
}

std::vector<cv::Mat> VideoStream::getFrames()
{
    return videoFrames;
}

std::string VideoStream::openFile(std::string fileName)
{
    capture.open(fileName);
    if (!capture.isOpened()) {
            return "Could not open reference ";
        }
    MainWindow::setTotalFrames(int(capture.get(cv::CAP_PROP_FRAME_COUNT)));
    MainWindow::setFps(int(capture.get(cv::CAP_PROP_FPS)));

    return fileName+ " loaded.";
}
