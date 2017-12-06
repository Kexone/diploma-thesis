#ifndef TRAINHOG_H
#define TRAINHOG_H

#include <opencv2/opencv.hpp>
#include <vector>


class TrainHog
{
protected:
    void extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst);

private:
    void trainSvm(cv::Mat &trainMat, const std::vector< int > &labels);
    void convertSamples2Mat(const std::vector< cv::Mat > &trainSamples, cv::Mat &trainData );
	void saveLabeledMat(cv::Mat data, std::vector< int > labels);


    std::string classifierName;
	cv::Size pedestrianSize;
    int blockSize = 16;
    int cellSize = 8;
    int strideSize = 8;

	int maxIterations;
	int termCriteria;
	int kernel;
	int type;
	double epsilon;
	double coef0;
	int degree;
	double gamma;
	double nu;
	double p;
	double c;

public:
	TrainHog();
	TrainHog(int maxIterations, int termCriteria, int kernel, int type, double epsilon, double coef0,
		int degree, double gamma, double nu, double p, double c, std::string classifierName);
	void printSettings();
	void trainFromMat(std::string path, std::string labelsPath);
    void train(std::string posSamples, std::string negSamples, bool saveData);
	cv::Size getPedSize();
};

#endif // TRAINHOG_H
