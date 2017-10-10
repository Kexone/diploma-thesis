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
    MediaFile();
    std::vector<cv::Mat> getFrames();
    std::string openFile(std::vector<std::string> mediaList);

private:
    std::vector<cv::Mat> origFrames;
    bool openImage(std::vector<std::string> mediaList);
};

#endif // MEDIAFILE_H
