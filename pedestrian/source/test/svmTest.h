#ifndef SVMTEST_H
#define SVMTEST_H

class SvmTest
{
private:
	void iterateValues();
	void print2File(int actualIter);
	void loadMats(std::string &samplesPath, std::vector< cv::Mat > &samples);
	void initResultFile();

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

	int nTruePos;
	int nFalsePos;
	int nTrueNeg;
	int nFalseNeg;

	int iterChange;
	double parChange;
	int maxIterTest;
	int typeTest;

public:
	SvmTest();
	void runSvmTest();
};

#endif // SVMTEST_H
