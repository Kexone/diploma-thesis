#include "mediafile.h"
#include "mainwindow.h"
MediaFile::MediaFile(bool isVideo)
{
    this->isVideo = isVideo;
}

MediaFile::~MediaFile() {
    //delete origPictures;
    //delete grayedPictures;
    delete this;
}

std::vector<cv::Mat> MediaFile::getFrames()
{
    return origFrames;
}

std::string MediaFile::openFile(std::vector<std::string> mediaList)
{
    if(!isVideo) {
        if(openImage(mediaList)) {
            return std::to_string(mediaList.size()) + " file(s) loaded.";
        }
        else
            return "Error in loading images";
    }
    else {
        if(openVideo(mediaList[0])) {
            return mediaList[0] + " loaded.";
        }
        else
            return "Could not open reference ";
    }
}

bool MediaFile::openVideo(std::string fileName)
{
    capture.open(fileName);
    if (!capture.isOpened()) {
        return false;
    }
    cv::Mat temp;
    for( ;; ){
        capture >> temp;
        if(temp.empty())
            break;
        origFrames.push_back(temp.clone());
    }
    MainWindow::setTotalFrames(origFrames.size());
    MainWindow::setFps(int(capture.get(cv::CAP_PROP_FPS)));
    capture.release();
    return true;
}

bool MediaFile::openImage(std::vector<std::string> mediaList)
{
    origFrames.clear();
    for(int i = 0; i < mediaList.size(); i++) {
        std::string fileLocation = mediaList[i];
        cv::Mat temp, grayTemp;
        //temp =
        //cv::cvtColor(temp, grayTemp, CV_BGR2GRAY);
        origFrames.push_back(cv::imread(fileLocation,CV_LOAD_IMAGE_COLOR));
        //grayedPictures.push_back(grayTemp);
    //    cv::imshow("test"+i, temp);
      //  cv::imshow("testGray"+i, grayTemp);
    }
    return true;
}
