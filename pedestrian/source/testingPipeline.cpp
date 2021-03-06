﻿#include "testingPipeline.h"


TestingPipeline::TestingPipeline(std::string testingFile)
{
	std::ifstream file;
	std::string video,sett, svm;
	int typeAlg;
	file.open(testingFile);
	if (file.is_open()) {
		while (!file.eof()) {
			video = "";
			file >> video >> sett >> svm >> typeAlg ;
			if (!video.empty()) {
				_videos2Test.push_back(video);
				_settings.push_back(sett);
				_typeAlg.push_back(typeAlg);
				_svms2Test.push_back(svm);
			}
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
	std::string algNames[] = {"", "HOG", "MOG + HOG", "FHOG", "FHOG + MOG"};
	fs << "\n{Název videa}        & Typ algoritmu 	& FPS & Délka detekce[s] & TP 		& FN 	& FP 	& F1 skóre[\\%] \\\\  \\hline" << std::endl;
	for (int i = 0; i < _videos2Test.size(); i++)
	{
		std::map<std::string, float> results;

		Settings::getSettings(_settings[i]);
		Pipeline pip = Pipeline(_svms2Test[i], _typeAlg[i]);
		Utils::setEvaluationFiles(_videos2Test[i]);
		std::cout << algNames[_typeAlg[i]] << " on " << _videos2Test[i] << " with " << _svms2Test[i] << std::endl;
		auto startTime = std::chrono::high_resolution_clock::now();
		pip.execute(_videos2Test[i]);
		auto endTime = std::chrono::high_resolution_clock::now();
		double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();

		results = pip.evaluate();
		if (_svms2Test[i] == "default")
			fs << "\t & " << " default " << algNames[_typeAlg[i]] << " & \t";
		else
			fs << "\t & " << algNames[_typeAlg[i]] << " & \t";
		saveResults(fs, results, time, true, _typeAlg[i]);
		results.clear();
	}
}

void TestingPipeline::saveResults(std::ofstream &file, std::map<std::string, float> results, double time, bool print, int type)
{
	if (print)	{
	//	std::cout << "FPS: " << VideoStream::fps << "." << std::endl;
		std::cout << "ALG FPS: " << VideoStream::totalFrames / (static_cast<float>(time/1000)) << "." << std::endl;
	//	std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
	//	std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s." << std::endl;
		std::cout << "Detection took " << static_cast<float>(time) << "ms." << std::endl;
	}

	// name & ALG FPS & Detection took & TP & FN & FP & F1
	file << VideoStream::totalFrames / (static_cast<float>(time)) << " & " << static_cast<float>(time) << " & " <<
		results["tp"] << " & \t" << results["fn"] << " & \t" << results["fp"] << " & \t" << results["f1"];
		file << "\t \\\\ \\cline{2-8}  " << std::endl;
}

