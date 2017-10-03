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
    cv::Mat temp;
    for( ;; ){
        capture >> temp;
        if(temp.empty())
            break;
        videoFrames.push_back(temp);
    }
    MainWindow::setTotalFrames(videoFrames.size());
    MainWindow::setFps(int(capture.get(cv::CAP_PROP_FPS)));
    capture.release();
    return fileName+ " loaded.";
}
