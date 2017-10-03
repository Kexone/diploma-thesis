#ifndef MEDIAFILE_H
#define MEDIAFILE_H

#include <iostream>
#include <QStringList>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
class MediaFile
{
public:
    MediaFile(QStringList mediaList);
    ~MediaFile();

private:
    std::vector<cv::Mat> origPictures;
    std::vector<cv::Mat> grayedPictures;
};

#endif // MEDIAFILE_H
