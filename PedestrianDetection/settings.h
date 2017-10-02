#ifndef SETTINGS_H
#define SETTINGS_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


struct Settings
{
    float mogThreshold;
    cv::Size convexHullSize;
    cv::Size convexHUllDeviation;
    float hogThreshold;
    bool showVideoFrames;
    bool trainHog;
    int algorithm;
    int positiveFrames;

};

#endif // SETTINGS_H
