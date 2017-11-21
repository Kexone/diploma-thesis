/* 
 * 
 * Testing trained SVM  on samples data
 * 
 */

#include <iostream>
#include <fstream>
#include "../train/trainhog.h"
#include "svmTest.h"
#include "../alg/hog.h"

// The number of clock ticks per minute
#define CLOCKS_PER_MIN  ((clock_t)60000)
#define PARAMETER_NU  1
#define PARAMETER_C  2
#define PARAMETER_P  3

SvmTest::SvmTest()
{

}

void SvmTest::runSvmTest()
{
	std::string samplesPath = "samples/";
	std::string posSamples = samplesPath + "listPosMin.txt";
	std::string negSamples = samplesPath + "listNegMin.txt";
	std::string posTest = samplesPath + "listPosTestMin.txt";
	std::string negTest = samplesPath + "listNegTestMin.txt";
	std::string classifierName = "test.yml";

	std::vector< cv::Mat > posTestLst;
	std::vector< cv::Mat > negTestLst;
	int actualIter = 1;

	std::cout << "\n\n******************" << std::endl;
	std::cout << "***  SVM TEST  ***" << std::endl;
	std::cout << "******************" << std::endl;
	std::cout << "NUMBER OF REPEATS: ";
	std::cin >> maxIterTest;

	if(maxIterTest <= 0)
	{
		std::cout << "NUMBER OF ITERATIONS MUST BE GREATER THAN ZERO!\nEND" << std::endl;
		return;
	}
	std::cout << "SOFT (1) OR GROSS (2) REGRESSION TEST: ";
	std::cin >> typeTest;

	if(typeTest == 1)
	{
		iterChange = 50;
		parChange = 0.001;
	}
	else
	{
		iterChange = 100;
		parChange = 0.01;
	}

	std::cout << "ITERATE ALL AT ONCE (1) \nITERATION ONE BY ONE (2) \nONLY ITERATION(3)\nCHOOSE TYPE: ";
	std::cin >> typeIncr;
	if (typeIncr == 2) {
		std::cout << "TESTING WILL TAKE PLACE MAX ITER * VARIABLES (" << maxIterTest * 3 << " ITER)" << std::endl;
		maxIterTest *= 3;
	}
	else if (typeIncr == 3)
	{
		std::cout << "NUMBER OF ITERATIONS PER CYCLE:";
		std::cin >> iterChange;
	}
	loadMats(posTest, posTestLst);
	loadMats(negTest, negTestLst);
	initResultFile();

	std::cout << std::endl  << "*** TESTING HAS STARTED ***" << std::endl << std::endl;

	
	this->maxIterations = 300;
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
		nTruePos = 0;
		nFalsePos = 0;
		nTrueNeg = 0;
		nFalseNeg = 0;

		std::cout << std::endl << actualIter << ". ITERATION OF TESTING" << std::endl;

		TrainHog th = TrainHog(maxIterations, termCriteria, kernel, type, epsilon, coef0, degree, gamma, nu, p, c, classifierName);

		trainTime = clock();
		th.fillVectors(posSamples);
		th.fillVectors(negSamples, true);
		th.train(false);
		trainTime = clock() - trainTime;

		Hog h = Hog(classifierName);

		std::cout << " << TESTING SVM >>" << std::endl;

		classTime = clock();
		h.detect(posTestLst, nTruePos, nFalsePos);
		h.detect(negTestLst, nTrueNeg, nFalseNeg, false);
		classTime = clock() - classTime;

		std::cout << "POS DETECTION [T/F] " << nTruePos << "/" << nFalsePos << std::endl;
		std::cout << "NEG DETECTION [T/F] " << nTrueNeg << "/" << nFalseNeg << std::endl;

		print2File(actualIter);
		if(typeIncr == 1)	incrementValues();
		else if(typeIncr == 2)
		{
			int par =(maxIterTest / (float)(maxIterTest / 3));
			if (par == PARAMETER_NU)
			{
				this->nu += parChange;
				this->p = 0.1;
				this->c = 0.1;
			}
			if (par == PARAMETER_C)
			{
				this->nu = 0.1;
				this->p = 0.1;
				this->c += parChange;
			}
			if (par == PARAMETER_P)
			{
				this->nu = 0.1;
				this->p += parChange;
				this->c = 0.1;
			}
			maxIterations += iterChange;
		}
		else if(typeIncr == 3)	maxIterations += iterChange;
		actualIter++;
	}
}


void SvmTest::initResultFile()
{
	std::string incrType = "SOFT";
	std::string iterType = "SINGLE";
	if (typeTest == 2) incrType = "GROSS";
	else if (typeIncr == 2) iterType = "ALL AT ONCE";
	else if (typeIncr == 3) iterType = "ONLY ITERATION";

	std::ofstream file;
	file.open("result.txt", std::ios::app);

	file << "__________________________________" << std::endl;
	file << "\t\t******************" << std::endl;
	file << "\t\t***  SVM TEST  ***" << std::endl;
	file << "\t\t******************" << std::endl;
	file << "NUMBER OF REPEATS: " << maxIterTest << std::endl;
	file << "REGRESSION TYPE: " << incrType << std::endl;
	file << "INCREMENT TYPE: " << iterType << std::endl;
	file << "__________________________________" << std::endl;

	file.close();
}

void SvmTest::print2File(int actualIter)
{
	std::ofstream file;
	file.open("result.txt", std::ios::app);

	file << "\n\t<< START" << actualIter << ".ITERATION>>" << std::endl << std::endl;
	file << "\t__SVM SETTINGS__" << std::endl;
	file << "\tMAX ITER: " << this->maxIterations << std::endl;
	file << "\tTERM CRIT: " << this->termCriteria << std::endl;
	file << "\tKERNEL: " << this->kernel << std::endl;
	file << "\tTYPE SVM: " << this->type << std::endl;
	file << "\tEPSILON: " << this->epsilon << std::endl;
	file << "\tCOEF0:" << this->coef0 << std::endl;
	file << "\tDEGREE: " << this->degree << std::endl;
	file << "\tGAMMA: " << this->gamma << std::endl;
	file << "\tNU: " << this->nu << std::endl;
	file << "\tP: " << this->p << std::endl;
	file << "\tC: " << this->c << std::endl;
	file << "\tTRAIN TIME: " << static_cast<float>(trainTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl;

	file << "\t__SVM RESULTS__" << std::endl;
	file << "POS GOOD: " << nTruePos << " POS BAD: " << nFalsePos << std::endl;
	file << "NEG GOOD: " << nTrueNeg << " NEG BAD: " << nFalseNeg << std::endl;
	file << "CLASS TIME: " << static_cast<float>(classTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl ;
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

		frame = cv::imread(oSample, CV_32FC3);
		cv::resize(frame, frame, cv::Size(48, 96));
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		samples.push_back(frame.clone());
	}
}

void SvmTest::incrementValues()
{
	maxIterations += iterChange;
	this->nu += parChange;
	this->p += parChange;
	this->c += parChange;
}
