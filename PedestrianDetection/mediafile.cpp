#include "mediafile.h"
#include "mainwindow.h"
MediaFile::MediaFile()
{
}

MediaFile::~MediaFile() {
    delete this;
}

std::vector<cv::Mat> MediaFile::getFrames()
{
    return origFrames;
}

std::string MediaFile::openFile(std::vector<std::string> mediaList)
{
        if(openImage(mediaList))
            return std::to_string(mediaList.size()) + " file(s) loaded.";
        else
            return "Error in loading images";
}

bool MediaFile::openImage(std::vector<std::string> mediaList)
{
    origFrames.clear();
    for(uint i = 0; i < mediaList.size(); i++) {
        std::string fileLocation = mediaList[i];
        origFrames.push_back(cv::imread(fileLocation,CV_LOAD_IMAGE_COLOR));
    }
    return true;
}
