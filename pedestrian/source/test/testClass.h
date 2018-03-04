#ifndef TESTCLASS_H
#define TESTCLASS_H

#include <iostream>
#include "diffEvoTest.h"
#include <sstream>
#include "svmTest.h"

/**
 *  class TestClass
 *  
 *  Class for testing classificator
 */
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

	void crossTestingSvm();
	void crossTestingDlibSvm();
	void testingSvm();
	void evaluate(std::string groundTruthFile, std::string resultsFilePath);
	
	/**
	 * @brief Runs the random tests
	 * Testing is going in three way: increment all params in one iterate, increment one by one or increment only max iteration (param to svm)
	 * In all ways can choosed the soft or gross iteration
	 */
	void randomTest();

	/**
	 * @brief Testing works with differential evolution algorithm
	 */
	void diffEvoTest();

	void incrementSvmValues(int typeIncr, int maxRepTest);

	/**
	 * @brief Testing in three nested cycles, testing svm classificator c_svc type (max iter, gamma, param c)
	 */
	void iterationCycle();

	/**
	 * @brief Initiate log to file
	 * Log is set by the test and sended to testing case
	 *
	 */
	void initLog(int typeTest, int typeIncr = 0, int maxRepeatTest = 0);

};

#endif // TESTCLASS_H