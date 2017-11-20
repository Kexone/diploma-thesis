/* 
 * 
 * Testing trained SVM 
 * 
 */

#include <iostream>
#include <fstream>
#include "../train/trainhog.h"
#include "svmTest.h"
#include "../alg/hog.h"

// The number of clock ticks per minute
#define CLOCKS_PER_MIN  ((clock_t)60000)

void print2File();
void iterateValues();

void SvmTest::runSvmTest()
{
	int maxIterTest;
	std::cout << "\n\n******************" << std::endl;
	std::cout << "***  SVM TEST  ***" << std::endl;
	std::cout << "******************" << std::endl;
	std::cout << "COUNT OF ITERATION: ";
	std::cin >> maxIterTest;
	if(maxIterTest <= 0)
	{
		std::cout << "COUNT OF ITERATION MUST BE GREATER THEN ZERO!\nEND" << std::endl;
		return;
	}
	std::string posSamples = "listPosMin.txt";
	std::string negSamples = "listNegMin.txt";
	std::string posTest = "listTestPos.txt";
	std::string negTest = "listTestNeg.txt";
	std::string classifierName = "test.yml";
	
	std::vector< cv::Mat > posTestLst;
	std::vector< cv::Mat > negTestLst;
	loadMats(posSamples, posTestLst);
	loadMats(negSamples, negTestLst);
	std::cout << "*** TESTING HAS STARTED ***" << std::endl << std::endl;

	int actualIter = 1;
	

	this->maxIterations = 3300;
	this->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	this->kernel = cv::ml::SVM::LINEAR;
	this->type = cv::ml::SVM::NU_SVC;
	this->epsilon = 1.e-6;
	this->coef0 = 0.0;
	this->degree = 3;
	this->gamma = 0.1;
	this->nu = 0.1;
	this->p = 0.1;
	this->c = 0.1;
	while(actualIter != maxIterTest+1)
	{
		badPos = 0;
		goodPos = 0;
		badNeg = 0;
		goodNeg = 0;

		std::cout << actualIter << ". ITERATION OF TESTING" << std::endl;

		//TrainHog th = TrainHog(maxIterations,termCriteria,kernel,type,epsilon,coef0,degree,gamma,nu,p,c,classifierName);

		trainTime = clock();
		//th.fillVectors(posSamples);
		//th.fillVectors(negSamples, true);
		//th.train(false);
		trainTime = clock() - trainTime;

		Hog h = Hog(classifierName);

		classTime = clock();
		h.detect(posTestLst,goodPos,badPos);
		h.detect(negTestLst,goodNeg, badNeg, false);
		classTime = clock() - classTime;

		print2File(actualIter);
		iterateValues();

		actualIter++;
	}
}

void SvmTest::print2File(int actualIter)
{
	std::ofstream file;
	file.open("result.txt", std::ios::app);

	file << "\n\t<< START" << actualIter << ".ITERATION>>" << std::endl;
	file << "\t__SVM SETTINGS__" << std::endl;
	file << "MAX ITER: " << this->maxIterations << std::endl;
	file << "TERM CRIT: " << this->termCriteria << std::endl;
	file << "KERNEL: " << this->kernel << std::endl;
	file << "TYPE SVM: " << this->type << std::endl;
	file << "EPSILON: " << this->epsilon << std::endl;
	file << "COEF0:" << this->coef0 << std::endl;
	file << "DEGREE: " << this->degree << std::endl;
	file << "GAMMA: " << this->gamma << std::endl;
	file << "NU: " << this->nu << std::endl;
	file << "P: " << this->p << std::endl;
	file << "C: " << this->c << std::endl;
	file << "TRAIN TIME: " << static_cast<float>(trainTime / CLOCKS_PER_MIN ) << " MIN" << std::endl;

	file << "\n\t__SVM RESULTS__" << std::endl;
	file << "POS GOOD: " << badPos << " POS BAD: " << goodPos  << std::endl;
	file << "NEG GOOD: " << badNeg << " NEG BAD: " << goodNeg  << std::endl;
	file << "CLASS TIME: " << static_cast<float>(classTime / CLOCKS_PER_MIN) << " MIN" << std::endl;
	file << "\t<< END" << actualIter << ".ITERATION>>" << std::endl;

	file.close();
}

void SvmTest::loadMats(std::string& samplesPath, std::vector<cv::Mat>& samples)
{
	assert(!samplesPath.empty());

	cv::Mat frame;
	std::fstream sampleFile(samplesPath);
	std::string oSample;
	while (sampleFile >> oSample) {

		//	frame = cv::imread(oSample, CV_32FC3);
		samples.push_back(cv::imread(oSample, CV_32FC3));
	}
}

void SvmTest::iterateValues()
{
	
}

SvmTest::SvmTest()
{
	
}
