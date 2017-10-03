#ifndef HOG_H
#define HOG_H
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/tracking.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/video/background_segm.hpp>
#include <opencv2/bgsegm.hpp>
#include <iostream>

class Hog
{
public:
    Hog();
    //std::vector<std::vector<cv::Rect>> detect(std::vector<CroppedImage>& frame);
    std::vector<std::vector<cv::Rect>> detect(std::vector<cv::Mat> frames);
    //cv::Mat resizeImage(const cv::Mat img, cv::Size target_size);
private:
    cv::HOGDescriptor hog;
};

#endif // HOG_H
