#ifndef TRAINHOG_H
#define TRAINHOG_H

#include <opencv2/opencv.hpp>
#include <vector>


class TrainHog
{
private:
    void extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst);
    void trainSvm(cv::Mat &trainMat, const std::vector< int > &labels);
    void convertSamples2Mat(const std::vector< cv::Mat > &trainSamples, cv::Mat &trainData );
	void saveMatWithLabs(cv::Mat data);
    std::vector< cv::Mat > posSamples;
    std::vector< cv::Mat > negSamples;
    std::vector< int > labels;
    std::string classifierName = "96_48_16_8_8_9_01.yml";
	cv::Size pedestrianSize;
    int blockSize = 16;
    int cellSize = 8;
    int strideSize = 8;

public:
    TrainHog();
    void fillVectors(std::string &samplesListPath, bool isNeg = false);
	void trainFromMat(std::string path, std::string labelsPath);
    void train(bool saveData);
};

#endif // TRAINHOG_H
