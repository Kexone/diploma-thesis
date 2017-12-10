#ifndef TESTCLASS_H
#define TESTCLASS_H

#include <iostream>
#include "diffEvoTest.h"
#include <sstream>
#include "svmTest.h"

class TestClass
{
public:
	TestClass();
	void initTesting();

private:

	int maxIterations = 300;
	float parChange;
	int iterChange;
	double nu;
	double p;
	double c;

	std::stringstream ss;

	void testingSvm();
	void randomTest(int typeTest);
	void diffEvoTest();
	void incrementSvmValues(int typeIncr, int maxRepTest);
	void iterationCycle();
	void initLog(int typeTest, int typeIncr = 0, int maxRepeatTest = 0);

};

#endif // TESTCLASS_H