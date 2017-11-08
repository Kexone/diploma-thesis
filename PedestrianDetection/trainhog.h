#ifndef TRAINHOG_H
#define TRAINHOG_H
#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

class TrainHog
{
private:
    void extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst);
    void trainSvm(cv::Mat &trainMat, const std::vector< int > &labels);
    void convertSamples2Mat(const std::vector< cv::Mat > &trainSamples, cv::Mat &trainData );
    std::vector< cv::Mat > posSamples;
    std::vector< cv::Mat > negSamples;
    std::vector< int > labels;
    std::string classifierName = "96_16_8_8_9_01.yml";
    int winSize = 96;
    int blockSize = 16;
    int cellSize = 8;
    int strideSize = 8;

public:
    TrainHog();
    void fillVectors(std::vector<std::string> &samplesListPath, bool isNeg = false);
    bool train();
};

#endif // TRAINHOG_H
