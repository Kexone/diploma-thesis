#ifndef CROPPEDIMAGE_H
#define CROPPEDIMAGE_H

#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
struct CroppedImage
{
public:
    int id;
    cv::Mat croppedImg;
    cv::Size defaultSize;
    int offsetX;
    int offsetY;

    CroppedImage(int i, cv::Mat frame, cv::Rect cropping) : id(i), croppedImg(frame(cropping)),
        defaultSize(cv::Size(frame.cols, frame.rows)), offsetX(cropping.x), offsetY(cropping.y) {}
};

#endif // CROPPEDIMAGE_H
