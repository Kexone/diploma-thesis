#ifndef MEDIAFILE_H
#define MEDIAFILE_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>


class MediaFile
{
public:
    MediaFile(bool isVideo);
    ~MediaFile();
    std::vector<cv::Mat> getFrames();
    std::string openFile(std::vector<std::string> mediaList);

private:
    std::vector<cv::Mat> videoFrames;
    cv::VideoCapture capture;
    bool isVideo;
    std::vector<cv::Mat> origFrames;
    std::vector<cv::Mat> grayedPictures;
    bool openImage(std::vector<std::string> mediaList);
    bool openVideo(std::string fileName);
};

#endif // MEDIAFILE_H
