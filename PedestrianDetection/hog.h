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
#include <opencv2/ml.hpp>
#include "croppedimage.h"
#include <iostream>

class Hog
{
public:
    Hog();
    std::vector<std::vector<cv::Rect>> detect(std::vector<CroppedImage>& frame);
    std::vector<cv::Rect> detect(cv::Mat frame);
private:
    void getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector );
    cv::HOGDescriptor hog;
};

#endif // HOG_H
