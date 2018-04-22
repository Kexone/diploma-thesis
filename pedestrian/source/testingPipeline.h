#ifndef TESTING_PIPELINE_H
#define TESTING_PIPELINE_H
#include <string>
#include "pipeline.h"
#include "utils/utils.h"
#include <chrono>
#include <ctime> 
class TestingPipeline
{
public:
	TestingPipeline() {};
	TestingPipeline(std::string testingFile);
	void execute();

private:
	std::vector < std::string > _svms2Test;
	std::vector < std::string > _videos2Test;
	std::vector < std::string > _settings;
	std::vector <  int > _typeAlg;
	void saveResults(std::ofstream &file, std::map<std::string, int> results, double time, bool print = false);
};

#endif // TESTING_PIPELINE_H
