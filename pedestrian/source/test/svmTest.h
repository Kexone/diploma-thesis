#ifndef SVMTEST_H
#define SVMTEST_H
#include <iostream>
#include <sstream>
#include <fstream>
#include "../train/trainhog.h"
#include "../alg/hog.h"

/**
 *  class SvmTest
 *  
 *  Testing OpenCV SVM classificator 
 */
class SvmTest
{
private:
	/**
	 * @brief Saving params and results to file
	 * 
	 * @param currentTestNumb This param represent current testing round and is required for separete iteration in file
	 * @param valuation 1D array of T/F states from detector
	 */
	void print2File(int currentTestNumb, int *valuation);

	void loadMats(std::string samplesPath, std::vector< cv::Mat > &lst);

	
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

	clock_t trainTime;
	clock_t classTime;

	float accuracy;

	std::string classifierName;
	std::string posSamplesMin;
	std::string negSamplesMin;

	cv::Mat trainMat;
	std::vector < int > labels;
	std::vector< cv::Mat > posTestLst;
	std::vector< cv::Mat > negTestLst;


public:
	SvmTest();
	/**
	 * @brief Sets params for train SVM
	 * 
	 * @param maxIter maximum iteration for train
	 * @param nu parameter nu (NU_SVC, ONE_CLASS, NU_SVR)
	 * @param c parameter c (C_SVC, EPS_SVR, NU_SVR)
	 * @param p parameter p for train (EPS_SVR)
	 */
	void setParams(int maxIter, double nu, double c, double p);

	/**
		* @brief Sets params for train SVM. This method is default for train SVM type C_SVC
		*
		* @param maxIter maximum iteration for train
		* @param c parameter c(C_SVC, EPS_SVR, NU_SVR)
		* @param gamma parameter gamma for train (POLY, RBF, SIGMOID, CHI2)
	*/
	void setParams(int maxIter, double c, double gamma);

	/**
	 * @brief Preprocessing of testing. This method loads the training set and sends it to train HOG class which get back train matrix
	 * 
	 */
	void preprocessing();

	/**
	 * @brief Firstly sets the params of SVM, next step is training
	 * after train SVM, calls the HOG detection function where calc the accuracy and returns T/F states
	 * subsequently calc the accuracy and save results and params to the file
	 */
	float process();

	/**
	 * @brief Initializes the result file from string stream
	 * 
	 * @param ss string stream contains the int settings (name, iteration etc.)
	 */
	static void initResultFile(std::stringstream &ss);
};

#endif // SVMTEST_H
