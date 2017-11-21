#ifndef SVMTEST_H
#define SVMTEST_H
#include "../../3dparty/de/DifferentialEvolution.h"

class SvmTest : public de::IOptimizable
{
private:
	void incrementValues();
	void print2File(int actualIter);
	void loadMats(std::string &samplesPath, std::vector< cv::Mat > &samples);
	void initResultFile();
	void process();
	double EvaluteCost(std::vector<double> inputs) const override;
	unsigned int NumberOfParameters() const override;	
	std::vector<Constraints> GetConstraints() const override;

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

	std::string classifierName;
	std::string posSamples;
	std::string negSamples;
	std::string posTest;
	std::string negTest;
	std::vector< cv::Mat > posTestLst;
	std::vector< cv::Mat > negTestLst;
	int actualIter;

	float accuracy;

	int nTruePos;
	int nFalsePos;
	int nTrueNeg;
	int nFalseNeg;

	int iterChange;
	double parChange;
	int maxIterTest;
	int typeTest;
	int typeIncr;

	unsigned int m_dim;

public:
	SvmTest();
	SvmTest(unsigned int dims)
	{
		m_dim = dims;
	}
	void runSvmTest();
};

#endif // SVMTEST_H
