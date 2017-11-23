﻿#pragma once
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
	float  iterChange;
	double nu;
	double p;
	double c;

	std::stringstream ss;

	void testingSvm();
	void randomTest(int typeTest);
	void diffEvoTest();
	void incrementSvmValues(int typeIncr, int maxRepTest);

	void initLog(int typeTest, int typeIncr = 0, int maxRepeatTest = 0);
	void values2Log();

};
