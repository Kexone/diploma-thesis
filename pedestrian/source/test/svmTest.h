#ifndef SVMTEST_H
#define SVMTEST_H
#include <iostream>
#include <sstream>
#include <fstream>
#include "../train/trainhog.h"
#include "../alg/hog.h"
class SvmTest
{
private:
	void print2File(int currentTestNumb, int *valuation, std::stringstream &ss);

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

	int nTruePos;
	int nFalsePos;
	int nTrueNeg;
	int nFalseNeg;
	std::string posSamplesMin;
	std::string negSamplesMin;

public:
	SvmTest();
	void setParams(int maxIter, double nu, double c, double p);
	float process(std::stringstream &ss);
	static void initResultFile(std::stringstream &ss);
};

#endif // SVMTEST_H
