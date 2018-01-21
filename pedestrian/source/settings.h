#ifndef SETTINGS_H
#define SETTINGS_H

#include <opencv2/highgui/highgui.hpp>

/**
 * Old settings
 * @TODO use it in code
 */
struct Settings
{
    static double mogThreshold;
    static int mogHistory;
    static cv::Size convexHullSize;
    static cv::Size convexHUllDeviation;
    static double hogThreshold;
    static bool trainHog;
    static int algorithm;
    static int positiveFrames;

    static bool showVideoFrames;
	static std::string nameFile;
	static std::string nameTrainedFile;

};

#endif // SETTINGS_H
