#include "testingPipeline.h"
#include <chrono>
#include <ctime>

TestingPipeline::TestingPipeline(std::string svmsPath, std::string videosPath)
{
	std::ifstream file;
	file.open(svmsPath);
	std::string line;
	if (file.is_open()) {
		while (!file.eof()) {
			file >> line;
			_svms2Test.push_back(line);
		}
	}
	file.close();

	file.open(videosPath);
	std::string line;
	if (file.is_open()) {
		while (!file.eof()) {
			file >> line;
			_videos2Test.push_back(line);
		}
	}
	file.close();	
}

void TestingPipeline::execute()
{
	std::ofstream fs;
	fs.open("armTestingResult.txt");
	std::time_t currTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	fs << "TESTING RESULT AT  " << std::ctime(&currTime) << std::endl;
	std::vector < std::string > videos;
	std::string algNames[] = { "HOG", "MOG + HOG", "FHOG", "MOG + FHOG" };
	for (auto video : videos) {

		fs << video << " FPS: " << VideoStream::fps << ". Video duration : " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) <<
			"s. Total frames: " << VideoStream::totalFrames << "." << std::endl << std::endl;

		for (size_t i = 0; i < _svms2Test.size(); i++) {
			for (size_t j = 0; j < _videos2Test.size(); j++) {

				std::map<std::string, int> results;
				Pipeline pip = Pipeline(_svms2Test[i], i);

				std::cout << algNames[i] << std::endl;
				auto startTime = std::chrono::high_resolution_clock::now();
				pip.execute(_videos2Test[j]);
				auto endTime = std::chrono::high_resolution_clock::now();
				double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();

				pip.evaluate(results);
				fs << algNames[i] << " & ";
				saveResults(fs, results, time);
			}
		}
		fs << std::endl << std::endl;
	}
}

void TestingPipeline::saveResults(std::ofstream &file, std::map<std::string, int> results, std::time_t time, bool print)
{
	if(print)
	{
		std::cout << "FPS: " << VideoStream::fps << "." << std::endl;
		std::cout << "ALG FPS: " << VideoStream::totalFrames / (static_cast<float>(time) / CLOCKS_PER_SEC) << "." << std::endl;
		std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
		std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s." << std::endl;
		std::cout << "Detection took " << static_cast<float>(time) / CLOCKS_PER_SEC << "s." << std::endl;
		std::cout << "Possibly detection: " << Pipeline::allDetections << " frames." << std::endl;
	}
	// name & ALG FPS & Detection took & TP & FN & FP & F1
	file << VideoStream::totalFrames / (static_cast<float>(time) / CLOCKS_PER_SEC) << " & " << static_cast<float>(time) / CLOCKS_PER_SEC << " & " <<
		results["tp"] << " & " << results["fn"] << " & " << results["fp"] << " & " << results["f1"] << " \\ " << std::endl;
}