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
	actualIter = 1;

	std::cout << "\n\n******************" << std::endl;
	std::cout << "***  SVM TEST  ***" << std::endl;
	std::cout << "******************" << std::endl;
	std::cout << "DIFFERENTIAL EVOLUTION (1)\nRANDOM TESTING (2)\nCHOOSE TYPE: ";
	std::cin >> typeTest;
	if(typeTest == 1)
	{
		typeTest = 3;
		initResultFile();
		SvmTest dims(3);
		de::DifferentialEvolution de(dims, 50);

		de.Optimize(1000, true);
	}
	std::cout << "NUMBER OF REPEATS: ";
	std::cin >> maxIterTest;

	if(maxIterTest <= 0)
	{
		std::cout << "NUMBER OF ITERATIONS MUST BE GREATER THAN ZERO!\nEND" << std::endl;
		return;
	}
	std::cout << "SOFT (1) OR GROSS (2) ITERATION: ";
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

	initResultFile();

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

	std::cout << std::endl  << "*** TESTING HAS STARTED ***" << std::endl << std::endl;

	while(actualIter != maxIterTest+1)
	{
		process();
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
	}
}


void SvmTest::initResultFile()
{
	std::string incrType = "SOFT";
	std::string iterType = "SINGLE";
	if (typeTest == 2) incrType = "GROSS";
	if (typeTest == 3) incrType = "DIFFERENTIAL EVOLUTION";
	else if (typeIncr == 2) iterType = "ALL AT ONCE";
	else if (typeIncr == 3) iterType = "ONLY ITERATION";

	std::ofstream file;
	file.open("result.txt", std::ios::app);

	file << "__________________________________" << std::endl;
	file << "\t\t******************" << std::endl;
	file << "\t\t***  SVM TEST  ***" << std::endl;
	file << "\t\t******************" << std::endl;
	if (typeTest != 3) {
		file << "NUMBER OF REPEATS: " << maxIterTest << std::endl;
		file << "REGRESSION TYPE: " << incrType << std::endl;
		file << "INCREMENT TYPE: " << iterType << std::endl;
	}
	else
	{
		file << "TYPE: " << incrType << std::endl;
	}
	file << "__________________________________" << std::endl;

	file.close();
}

void SvmTest::process()
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

	actualIter++;
	accuracy = static_cast<float>(nTruePos + nTrueNeg) / static_cast<float>(nTruePos + nTrueNeg + nFalsePos + nFalseNeg) ;
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
	file << "ACCURACY: " << accuracy << " %" << std::endl;
	file << "CLASS TIME: " << static_cast<float>(classTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl ;
	file << "\t<< END" << actualIter << ".ITERATION>>" << std::endl;

	file.close();
}

void SvmTest::incrementValues()
{
	maxIterations += iterChange;
	this->nu += parChange;
	this->p += parChange;
	this->c += parChange;
}


double SvmTest::EvaluteCost(std::vector<double> inputs) const
{
	const_cast<SvmTest*>(this)->maxIterations = 330;//inputs[3] * 1000;
	const_cast<SvmTest*>(this)->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	const_cast<SvmTest*>(this)->kernel = cv::ml::SVM::LINEAR;
	const_cast<SvmTest*>(this)->type = cv::ml::SVM::NU_SVC;
	const_cast<SvmTest*>(this)->epsilon = 1.e-6;
	const_cast<SvmTest*>(this)->coef0 = 0.0;
	const_cast<SvmTest*>(this)->degree = 0.1;
	const_cast<SvmTest*>(this)->gamma = 0.3;
	const_cast<SvmTest*>(this)->c = inputs[0];
	const_cast<SvmTest*>(this)->p = inputs[1];
	const_cast<SvmTest*>(this)->nu = inputs[2];
	const_cast<SvmTest*>(this)->process();
	const_cast<SvmTest*>(this)->print2File(const_cast<SvmTest*>(this)->actualIter);

	std::cout << const_cast<SvmTest*>(this)->accuracy << std::endl;
	return 1 - const_cast<SvmTest*>(this)->accuracy;
}

unsigned int SvmTest::NumberOfParameters() const
{
	return 3;
}

std::vector<de::IOptimizable::Constraints> SvmTest::GetConstraints() const
{
	std::vector<Constraints> constr;

	constr.push_back(Constraints(0.0, 1.0, true));
	constr.push_back(Constraints(0.0, 1.0, true));
	constr.push_back(Constraints(0.0, 1.0, true));
	//constr.push_back(Constraints(0.3, 1.0, true));
	return constr;
}