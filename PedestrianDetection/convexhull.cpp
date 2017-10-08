#include "convexhull.h"
#include "settings.h"
#include "mainwindow.h"


ConvexHull::ConvexHull()
{

}

ConvexHull::ConvexHull(cv::Mat src, cv::Mat src_gray, int thresh) {
    src.copyTo(this->src);
    src_gray.copyTo(this->src_gray);
    this->thresh = Settings::mogThreshold;
    int max_thresh = 255;
    //cv::createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);

}


/** @function thresh_callback */
std::vector<std::vector<cv::Rect>> ConvexHull::thresh_callback(int, void*)
{
    cv::RNG rng(12345);
    cv::Mat src_copy = src.clone();
    cv::Mat orig = src.clone();
    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    /// Detect edges using Threshold
    cv::cvtColor(src,src_gray, CV_BGR2GRAY);
    assert(!src_gray.empty());
    cv::threshold(src_gray, threshold_output, 180, 255, cv::THRESH_BINARY);
    /// Find contours
    assert(!threshold_output.empty());
    cv::findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    /// Find the convex hull object for each contour
    std::vector<std::vector<cv::Point> >hull(contours.size());
    for (uint i = 0; i < contours.size(); i++)
    {
        convexHull(cv::Mat(contours[i]), hull[i], false);
    }

    std::vector<std::vector<cv::Point>>filteredHulls;
    int minThresholdArea = 5 * 100 , maxThresholdArea = 400 * 400;

    for (uint i = 0; i < hull.size(); i++) {
        int minX = INT_MAX, minY = INT_MAX, maxY = 0, maxX = 0;

        for (auto &p : hull[i]) {
            if (p.x <= minX) minX = p.x;
            if (p.y <= minY) minY = p.y;
            if (p.x >= maxX) maxX = p.x;
            if (p.y >= maxY) maxY = p.y;
        }

        // Vypočítej obsah
        if ((maxX - minX) * (maxY - minY) > minThresholdArea && (maxX - minX) * (maxY - minY) < maxThresholdArea)
             {
            filteredHulls.push_back(hull[i]);
        }
    }


    /// Draw contours + hull results
    std::vector<std::vector<cv::Rect>> react(filteredHulls.size());
    cv::Mat drawing = cv::Mat::zeros(threshold_output.size(), CV_8UC3);
    for (uint i = 0; i < filteredHulls.size(); i++)
    {
        int minX = INT_MAX, minY = INT_MAX, maxY = 0, maxX = 0;
        for (auto &p : filteredHulls[i]) {
            if (p.x <= minX) minX = p.x;
            if (p.y <= minY) minY = p.y;
            if (p.x >= maxX) maxX = p.x;
            if (p.y >= maxY) maxY = p.y;
        }
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(threshold_output, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        cv::drawContours(threshold_output, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
       // drawContours(drawing, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        cv::Rect rectangle = cv::Rect(cv::Point(minX, minY), cv::Point(maxX, maxY));
        react[i].push_back(rectangle);
    }

    /// Show in a window
    imshow("Hull demo", threshold_output);
    src_copy.release();
    orig.release();
    threshold_output.release();
    return react;
}
