#ifndef CROPPEDIMAGE_H
#define CROPPEDIMAGE_H

#include "opencv2/imgcodecs.hpp"

/**
 * Struct CroppedImage
 * 
 * This struct serves for save clipping from frame and his offsets
 */
struct CroppedImage
{
public:
    int id;
    cv::Mat croppedImg;
    int offsetX;
    int offsetY;

    CroppedImage(int i, cv::Mat frame, cv::Rect cropping) : id(i), croppedImg(frame(cropping)),
        offsetX(cropping.x), offsetY(cropping.y) {}
};

#endif // CROPPEDIMAGE_H
