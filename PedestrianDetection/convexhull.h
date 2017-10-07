#ifndef CONVEXHULL_H
#define CONVEXHULL_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

class ConvexHull
{
public:
    ConvexHull();
    ConvexHull(cv::Mat src, cv::Mat src_gray, int thresh);
    std::vector<std::vector<cv::Rect>> thresh_callback(int, void*);

private:
    cv::Mat src;
    cv::Mat src_gray;
    double thresh;
};

#endif // CONVEXHULL_H


