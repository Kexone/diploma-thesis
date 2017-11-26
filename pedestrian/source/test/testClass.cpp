﻿#include "testClass.h"

#define PARAMETER_NU  1
#define PARAMETER_C  2
#define PARAMETER_P  3



TestClass::TestClass()
{
	
}

void TestClass::initTesting()
{
	testingSvm();
}

void TestClass::testingSvm()
{
	int typeTest;
	std::cout << "\n\n******************" << std::endl;
	std::cout << "***  SVM TEST  ***" << std::endl;
	std::cout << "******************" << std::endl;
	std::cout << "DIFFERENTIAL EVOLUTION (1)\nRANDOM TESTING (2)\nCHOOSE TYPE: ";
	std::cin >> typeTest;


	if (typeTest == 1)
	{
		initLog(3);
		SvmTest::initResultFile(ss);
		diffEvoTest();
	}
	else if (typeTest == 2)
		randomTest(typeTest);
}

void TestClass::randomTest(int typeTest)
{
	int maxRepeatTest, typeIncr;
	
	std::cout << "NUMBER OF REPEATS: ";
	std::cin >> maxRepeatTest;

	if (maxRepeatTest <= 0)
	{
		std::cout << "NUMBER OF ITERATIONS MUST BE GREATER THAN ZERO!\nEND" << std::endl;
		return;
	}
	std::cout << "SOFT (1) OR GROSS (2) ITERATION: ";
	std::cin >> typeTest;

	if (typeTest == 1)
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
		std::cout << "TESTING WILL TAKE PLACE MAX ITER * VARIABLES (" << maxRepeatTest * 3 << " ITER)" << std::endl;
		maxRepeatTest *= 3;
	}
	else if (typeIncr == 3)
	{
		std::cout << "NUMBER OF ITERATIONS PER CYCLE:";
		std::cin >> iterChange;
	}
	initLog(typeTest,typeIncr,maxRepeatTest);
	nu = 0.0;
	p = 0.0;
	c = 0.0;
	SvmTest svm;
	
	SvmTest::initResultFile(ss);

	std::cout << std::endl << "*** TESTING HAS STARTED ***" << std::endl << std::endl;

	while (maxRepeatTest != 0)
	{
		incrementSvmValues(typeIncr, maxRepeatTest);

		svm.setParams(maxIterations, nu, c, p);
		svm.process();

		maxRepeatTest--;
	}
}
void TestClass::incrementSvmValues(int typeIncr, int maxRepTest)
{
	if (typeIncr == 1)
	{
		maxIterations += iterChange;
		nu += parChange;
		p += parChange;
		c += parChange;
	}
	else if (typeIncr == 2)
	{
		int par = (maxRepTest / (float)(maxRepTest / 3));
		if (par == PARAMETER_NU)
		{
			nu += parChange;
			p = 0.1;
			c = 0.1;
		}
		if (par == PARAMETER_C)
		{
			nu = 0.1;
			p = 0.1;
			c += parChange;
		}
		if (par == PARAMETER_P)
		{
			nu = 0.1;
			p += parChange;
			c = 0.1;
		}
		maxIterations += iterChange;
	}
	else if (typeIncr == 3)	maxIterations += iterChange;


}

void TestClass::diffEvoTest()
{
	int dimsCount = 4;
	std::cout << "DIMENSIONS ARE DEFAULT SET TO 4." << std::endl;
	std::cout << "POPULATION IS DEFAULT SET TO 50." << std::endl;

	DiffEvoTest dims(dimsCount);
	de::DifferentialEvolution de(dims, 50);

	de.Optimize(1000, true);
}

void TestClass::initLog(int typeTest, int typeIncr, int maxRepeatTest)
{
	std::string incrType = "SOFT";
	std::string iterType = "SINGLE";
	std::string testName = "SVM";
	if (typeTest == 2) incrType = "GROSS";
	if (typeTest == 3) incrType = "DIFFERENTIAL EVOLUTION";
	else if (typeIncr == 2) iterType = "ALL AT ONCE";
	else if (typeIncr == 3) iterType = "ONLY ITERATION";


	ss << "__________________________________" << std::endl;
	ss << "\t\t******************" << std::endl;
	ss << "\t\t***  "<< testName <<" TEST  ***" << std::endl;
	ss << "\t\t******************" << std::endl;
	if (typeTest != 3) {
		ss << "NUMBER OF REPEATS: " << maxRepeatTest << std::endl;
		ss << "REGRESSION TYPE: " << incrType << std::endl;
		ss << "INCREMENT TYPE: " << iterType << std::endl;
	}
	else
	{
		ss << "TYPE: " << incrType << std::endl;
	}
	ss << "__________________________________" << std::endl;
}
