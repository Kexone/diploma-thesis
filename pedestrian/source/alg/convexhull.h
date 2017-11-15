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
    std::vector<std::vector<cv::Rect>> wrapObjects(cv::Mat src, cv::Mat src_gray);

private:
    cv::Mat src;
    cv::Mat src_gray;
    double thresh;
	int extension;
	void filterContours(std::vector< std::vector< cv::Point > > &hulls, std::vector< std::vector< cv::Point > > &filteredHulls);
};

#endif // CONVEXHULL_H


