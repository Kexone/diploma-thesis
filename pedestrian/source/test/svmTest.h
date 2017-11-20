#ifndef SVMTEST_H
#define SVMTEST_H

class SvmTest
{
private:
	void iterateValues();
	void print2File(int actualIter);
	void loadMats(std::string &samplesPath, std::vector< cv::Mat > &samples);

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
	int badPos;
	int goodPos;
	int badNeg;
	int goodNeg;

public:
	SvmTest();
	void runSvmTest();
};

#endif // SVMTEST_H
