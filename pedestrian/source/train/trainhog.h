#ifndef TRAINHOG_H
#define TRAINHOG_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "../utils/utils.h"
#include "../settings.h"

/**
 * class TrainHog
 * 
 * @brief training class for OpenCV HoG
 */
class TrainHog
{
private:

	/**
	* @brief Train SVM classificatior
	*
	* @param trainMat training matrix created from gradients
	* @param labels vector of labels to training data
	*/
    void trainSvm(cv::Mat &trainMat, const std::vector< int > &labels);

	/**
	* @brief This method convert samples to training matrix
	*
	* @param trainSamples vector of samples to train
	* @param trainData is output training matrix
	*/
    void convertSamples2Mat(const std::vector< cv::Mat > &trainSamples, cv::Mat &trainData );

	/**
	* @brief This method can save training matrix include labels
	*
	* @param data matrix data
	* @param labels labels for data
	*/
	void saveLabeledMat(cv::Mat data, std::vector< int > labels);

	void getSvmDetector(const cv::Ptr<cv::ml::SVM>& svm, std::vector< float > & hog_detector);
	/**
	 * @brief Function for showing calculated gradients on image
	 * From https://github.com/opencv/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/samples/cpp/train_HOG.cpp#L134
	 *
	 *@param color_origImg matrix
	 *@param descriptorValues vector of calculated descriptors
	 *@param size size of mat
	 */
	cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size);

    std::string classifierName;
	cv::Size pedestrianSize;
	cv::Ptr<cv::ml::SVM> svm;

    int blockSize;
    int cellSize;
    int strideSize;

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

	/**
	* @brief print settings for training SVM classificator
	*
	*/
	void printSettings();

	/**
	* @brief Training from existing train matrix
	* Tihs method allow to train classificator from pre trained matrix and her labels
	*
	* @param path path to matrix
	* @param labelsPath path to labels
	*/
	void trainFromMat(std::string path, std::string labelsPath);

	/**
	* @brief Training from existing train matrix
	* Tihs method allow to train classificator from pre trained matrix and her labels
	* It is only encapsulating for train SVM classificator
	*
	* @param trainMat training matrix
	* @param labels vector of labels
	*/
	void trainFromMat(cv::Mat trainMat, std::vector < int > labels);

	/**
	* @brief  This method load samples and prepare them for alone train classificator
	*
	* @saveData condition for save trained matrix and her labels
	* @trainTwice condition for double train (bootstraping)
	*/
    void train(bool saveData = false, bool trainTwice = false);

	/**
	* @brief Extractiong gradients from samples by one by one and stored in vector gradientLst
	*
	* @param samplesLst list of samples
	* @param gradientLst list of gradient (output)
	*/
	void extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst) const;

	/**
	* @brief This method is appropriate for testing SVM classification, do only train the matrix and prepare the labels
	*
	* @param trainMat training matrix
	* @param labels vector of labels
	* @param isDlib switcher between libraries cause to change filling numbers in labels
	*/
	void calcMatForTraining(cv::Mat &trainMat, std::vector < int > &labels, bool isDlib = false);

};

#endif // TRAINHOG_H
