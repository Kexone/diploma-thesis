#include "mediafile.h"

MediaFile::MediaFile(QStringList mediaList)
{
    for(int i = 0; i < mediaList.size(); i++) {
        std::string fileLocation = mediaList[i].toUtf8().constData();
        cv::Mat temp, grayTemp;
        temp = cv::imread(fileLocation,CV_LOAD_IMAGE_COLOR);
        cv::cvtColor(temp, grayTemp, CV_BGR2GRAY);
        origPictures.push_back(temp);
        grayedPictures.push_back(grayTemp);
    //    cv::imshow("test"+i, temp);
      //  cv::imshow("testGray"+i, grayTemp);
    }
}

MediaFile::~MediaFile() {
    //delete origPictures;
    //delete grayedPictures;
    delete this;
}
