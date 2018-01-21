#ifndef SETTINGS_H
#define SETTINGS_H

#include <opencv2/highgui/highgui.hpp>

/**
 * @brief Setting struct for program
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
