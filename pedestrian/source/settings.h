#ifndef SETTINGS_H
#define SETTINGS_H

#include <iostream>
#include <fstream>

/**
 * @brief Setting struct for program
 */
struct Settings
{
    static int mogHistory;
    static double mogThresh;
	static bool mogDetectShadows;
	static int cvxHullExtSize;
	static int cvxHullExtTimes;
	static double cvxHullThresh;
	static double cvxHullMaxValue;
  //  static cv::Size convexHullSize;
  //  static double hogThreshold;

	static cv::Size pedSize;
	static int blockSize;
	static int cellSize;
	static int strideSize;
	static int maxIterations;
	static int termCriteria;
	static int kernel;
	static int type;
	static double epsilon;
	static double coef0;
	static int degree;
	static double gamma;
	static double paramNu;
	static double paramP;
	static double paramC;


    static int algorithm;
    static int positiveFrames;

    static bool showVideoFrames;
	static std::string nameFile;
	static std::string nameTrainedFile;
	static std::string classifierName2Train;

	static std::string samplesPos;
	static std::string samplesNeg;
	static std::string samplesPosTest;
	static std::string samplesNegTest;

	static int dilationSize;
	static int erosionSize;

	static cv::Size hogBlurFilter;
	static double hogHitTreshold;
	static cv::Size hogWinStride;
	static cv::Size hogPadding;
	static double hogScale;
	static double hogFinalTreshold;
	static bool hogMeanshiftGrouping;
	static int hogGroupTreshold;
	static double hogEps;

	static double cropHogHitTreshold;
	static cv::Size cropHogWinStride;
	static cv::Size cropHogPadding;
	static double cropHogScale;
	static double cropHogFinalTreshold;
	static bool cropHogMeanshiftGrouping;
	static int cropHogGroupTreshold;
	static double cropHogEps;
	static int cropHogMinArea;

	static void getSettings(std::string pathFile)
	{
		std::fstream file;
		file.open(pathFile);

		std::string line;
		while (file >> line) {
			try {
				std::istringstream is_line(line);
				std::string key;
				if (std::getline(is_line, key, '=')) {
					std::string value;
					if (std::getline(is_line, value)) {
						if (key.compare("mogHistory") == 0)
							mogHistory = std::stoi(value.c_str());
						else if (key.compare("mogThresh") == 0)
							mogThresh = std::stod(value.c_str());
						else if (key.compare("mogDetectShadows") == 0)
							mogDetectShadows = std::stoi(value.c_str());
						else if (key.compare("cvxHullExtSize") == 0)
							cvxHullExtSize = std::stoi(value.c_str());
						else if (key.compare("cvxHullExtTimes") == 0)
							cvxHullExtTimes = std::stoi(value.c_str());
						else if (key.compare("cvxHullThresh") == 0)
							cvxHullThresh = std::stod(value.c_str());
						else if (key.compare("cvxHullMaxValue") == 0)
							cvxHullMaxValue = std::stod(value.c_str());
						else if (key.compare("pedSize") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							pedSize = cv::Size(value1, value2);
						}
						else if (key.compare("blockSize") == 0)
							blockSize = std::stoi(value.c_str());
						else if (key.compare("cellSize") == 0)
							cellSize = std::stoi(value.c_str());
						else if (key.compare("strideSize") == 0)
							strideSize = std::stoi(value.c_str());
						else if (key.compare("maxIterations") == 0)
							maxIterations = std::stoi(value.c_str());
						else if (key.compare("termCriteria") == 0)
							termCriteria = std::stoi(value.c_str());
						else if (key.compare("kernel") == 0)
							kernel = std::stoi(value.c_str());
						else if (key.compare("type") == 0)
							type = std::stoi(value.c_str());
						else if (key.compare("epsilon") == 0)
							epsilon = std::stod(value.c_str());
						else if (key.compare("coef0") == 0)
							coef0 = std::stod(value.c_str());
						else if (key.compare("degree") == 0)
							degree = std::stoi(value.c_str());
						else if (key.compare("gamma") == 0)
							gamma = std::stod(value.c_str());
						else if (key.compare("paramNu") == 0)
							paramNu = std::stod(value.c_str());
						else if (key.compare("paramP") == 0)
							paramP = std::stod(value.c_str());
						else if (key.compare("paramC") == 0)
							paramC = std::stod(value.c_str());
						else if (key.compare("samplesPosTest") == 0)
							samplesPosTest = value.c_str();
						else if (key.compare("samplesNegTest") == 0)
							samplesNegTest = value.c_str();
						else if (key.compare("samplesPos") == 0)
							samplesPos = value.c_str();
						else if (key.compare("samplesNeg") == 0)
							samplesNeg = value.c_str();
						else if (key.compare("classifierName2Train") == 0)
							classifierName2Train = value.c_str();
						else if (key.compare("dilationSize") == 0)
							dilationSize = std::stoi(value.c_str());
						else if (key.compare("erosionSize") == 0)
							erosionSize = std::stoi(value.c_str());
						else if (key.compare("hogBlurFilter") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							hogBlurFilter = cv::Size(value1, value2);
						}
						else if (key.compare("hogHitTreshold") == 0)
							hogHitTreshold = std::stod(value.c_str());
						else if (key.compare("hogWinStride") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							hogWinStride = cv::Size(value1, value2);
						}
						else if (key.compare("hogPadding") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							hogPadding = cv::Size(value1, value2);
						}
						else if (key.compare("hogScale") == 0)
							hogScale = std::stod(value.c_str());
						else if (key.compare("hogFinalTreshold") == 0)
							hogFinalTreshold = std::stod(value.c_str());
						else if (key.compare("hogMeanshiftGrouping") == 0)
							hogMeanshiftGrouping = std::stoi(value.c_str());
						else if (key.compare("hogGroupTreshold") == 0)
							hogGroupTreshold = std::stoi(value.c_str());
						else if (key.compare("hogEps") == 0)
							hogEps = std::stod(value.c_str());
						else if (key.compare("cropHogHitTreshold") == 0)
							cropHogHitTreshold = std::stod(value.c_str());
						else if (key.compare("cropHogWinStride") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							cropHogWinStride = cv::Size(value1, value2);
						}
						else if (key.compare("cropHogPadding") == 0) {
							int value1, value2;
							auto commaPos = value.find(',');
							value1 = stoi(value.substr(1, commaPos - 1));
							value2 = stoi(value.substr(commaPos + 1, value.length() - 1));
							cropHogPadding = cv::Size(value1, value2);
						}
						else if (key.compare("cropHogScale") == 0)
							cropHogScale = std::stod(value.c_str());
						else if (key.compare("cropHogFinalTreshold") == 0)
							cropHogFinalTreshold = std::stod(value.c_str());
						else if (key.compare("cropHogMeanshiftGrouping") == 0)
							cropHogMeanshiftGrouping = std::stoi(value.c_str());
						else if (key.compare("cropHogGroupTreshold") == 0)
							cropHogGroupTreshold = std::stoi(value.c_str());
						else if (key.compare("cropHogEps") == 0)
							cropHogEps = std::stod(value.c_str());
						else if (key.compare("cropHogMinArea") == 0)
							cropHogMinArea = std::stoi(value.c_str());
					}
				}
			}
			catch(const std::exception e)
			{
				std::cout << " Error parsing file!\n\n " << e.what() << std::endl;
			}
		}
		file.close();
	}
};

#endif // SETTINGS_H
