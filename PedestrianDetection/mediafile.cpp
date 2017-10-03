#include "mediafile.h"

MediaFile::MediaFile(QStringList mediaList)
{
    origPictures.clear();
    for(int i = 0; i < mediaList.size(); i++) {
        std::string fileLocation = mediaList[i].toUtf8().constData();
        cv::Mat temp, grayTemp;
        //temp =
        //cv::cvtColor(temp, grayTemp, CV_BGR2GRAY);
        origPictures.push_back(cv::imread(fileLocation,CV_LOAD_IMAGE_COLOR));
        //grayedPictures.push_back(grayTemp);
    //    cv::imshow("test"+i, temp);
      //  cv::imshow("testGray"+i, grayTemp);
    }
}

MediaFile::~MediaFile() {
    //delete origPictures;
    //delete grayedPictures;
    delete this;
}

std::vector<cv::Mat> MediaFile::getFrames()
{
   return origPictures;
}
