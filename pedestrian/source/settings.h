#ifndef SETTINGS_H
#define SETTINGS_H

#include <opencv2/highgui/highgui.hpp>


struct Settings
{
    static double mogThreshold;
    static int mogHistory;
    static cv::Size convexHullSize;
    static cv::Size convexHUllDeviation;
    static double hogThreshold;
    static bool showVideoFrames;
    static bool trainHog;
    static int algorithm;
    static int positiveFrames;

};

#endif // SETTINGS_H
