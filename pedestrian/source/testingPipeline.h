#ifndef TESTING_PIPELINE_H
#define TESTING_PIPELINE_H
#include <string>
#include "pipeline.h"

class TestingPipeline
{
public:
	TestingPipeline() {};
	TestingPipeline(std::string svmsPath, std::string videosPath);
	void execute();

private:
	std::vector < std::string > _svms2Test;
	std::vector < std::string > _videos2Test;
	void saveResults(std::ofstream &file, std::map<std::string, int> results, std::time_t time, bool print = false);
};

#endif // TESTING_PIPELINE_H
