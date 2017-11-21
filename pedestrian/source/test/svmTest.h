#ifndef SVMTEST_H
#define SVMTEST_H
#include "../../3dparty/de/DifferentialEvolution.h"

class SvmTest : public de::IOptimizable
{
private:
	void incrementValues();
	void print2File(int actualIter);

	std::vector< cv::Mat > loadMats(const std::string samplesPath)
	{
		assert(!samplesPath.empty());
		std::vector< cv::Mat > tmp;
		cv::Mat frame;
		std::fstream sampleFile(samplesPath);
		std::string oSample;
		while (sampleFile >> oSample) {

			frame = cv::imread(oSample, CV_32FC3);
			cv::resize(frame, frame, cv::Size(48, 96));
			cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

			tmp.push_back(frame.clone());
		}
		return tmp;
	}

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

	const std::string classifierName = "test.yml";
	const std::string posSamples = "samples/listPosMin.txt";
	const std::string negSamples = "samples/listNegMin.txt";
	const std::string posTest = "samples/listPosTestMin.txt";
	const std::string negTest = "samples/listNegTestMin.txt";
	const std::vector< cv::Mat > posTestLst = loadMats(posTest);
	const std::vector< cv::Mat > negTestLst = loadMats(negTest);
	int actualIter = 1;

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
