#ifndef MEDIAFILE_H
#define MEDIAFILE_H

#include <opencv2/highgui.hpp>
#include <vector>

/**
 * class MediaFile
 */
class MediaFile
{
public:
    MediaFile();
    std::vector<cv::Mat> getFrames();
    std::string openFile(std::vector< std::string > mediaList);

private:
    std::vector<cv::Mat> origFrames;
    bool openImage(std::vector< std::string > mediaList);
};

#endif // MEDIAFILE_H
