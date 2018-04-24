#include <iostream>
#include <cstdlib>
#include <csignal>
#include <opencv2/core/utility.hpp>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include "source/testingPipeline.h"
#include "source/train/combinedTrainHog.h"
#include "source/utils/extractorROI.h"
#include "source/utils/utils.h"
#include "source/test/testClass.h"
#include <fenv.h>
#pragma STDC FENV_ACCESS ON

///////////////////////
//					//
//////////////////////////////////////////	/////////////
//	 DECLARATION	 //
//////////////////////
namespace mainFun {
	void type(cv::CommandLineParser parser);
	void camera(cv::CommandLineParser parser);
	void video(cv::CommandLineParser parser);
	void image(cv::CommandLineParser parser);
	void extract(cv::CommandLineParser parser);
	void createSample(cv::CommandLineParser parser);
	/**
	* Print the results on screen
	* @param timer represent time of the duration of the algorithm
	*/
	void printResults(clock_t timer);
}


///////////////////////
//					//
///////////////////////////////////////////////////////
//		 MAIN		 //
//////////////////////

/*
* @TODO train HAAR, LBP cascade classificator
*/
std::string Settings::nameFile = "";
std::string Settings::nameTrainedFile = "";
bool Settings::showVideoFrames = false;

int main(int argc, char *argv[])
{
	const int originalRounding = fegetround();
	fesetround(FE_TOWARDZERO);
	if(argc < 2)	{
		std::cout << "\tType -help for help" << std::endl;
		return 0;
	}
	const cv::String keys =
		"{ help h ?           |                                |  print help message                       }"
		"{ alg	              |               1                |  alg type                                 }"
		"{ video v            |                                |  use video as input                       }"
		"{ image i            |                                |  use list of images as input              }"
		"{ camera c           |                                |  enable camera capturing                  }"
		"{ class svm          |            default             |  trained clasifier path                   }"
		"{ settings  st       |   data/settings/settings.txt   |  file with settings for app               }"
		"{ type  t            |                                |  type of alg (train, test)                }"
		"{ extract e          |                                |  extract ROI from videostream             }"
		"{ vizualize viz      |               0                |  show result in window                    }"
		"{ verbose            |               0                |  print information about train etc.       }"
		"{ createSample cs    |               0                |  creating samples from image              }"
		;
	

	cv::CommandLineParser parser(argc, argv, keys);

	Settings::getSettings(parser.get<std::string>("settings"));
	Settings::showVideoFrames =  parser.get<int>("vizualize")  == 1 ? true : false;

	parser.about("DIPLOMA THESIS - Pedestrian Detection v0.6");


	if (parser.has("help"))	{
		parser.printMessage();
		return 0;
	}
	if (parser.has("type"))	{
		mainFun::type(parser);
	}
	else if (parser.has("camera"))	{
		mainFun::camera(parser);
	}
	else if (parser.has("video"))	{
		mainFun::video(parser);
	}
	else if (parser.has("image"))	{
		mainFun::image(parser);
	}
	else if (parser.has("extract")) {
		mainFun::extract(parser);
	}
	else if (parser.has("createSample")) {
		mainFun::createSample(parser);
	}
	fesetround(originalRounding);
	 	return 0;
}


void mainFun::type(cv::CommandLineParser parser)
{
	std::string type = parser.get<std::string>("type");
	if (!type.compare("test"))	{
		TestClass tc;
		tc.initTesting();
	}
	int chosenType;
	if (!type.compare("train"))	{

		std::cout << "\n 1) openCV SVM train \n 2) combined train (extract features by opencv HOG and train by dlib SVM) \n";
		std::cout << " 3) dlib SVM train \nType of train : ";
		std::cin >> chosenType;

		if (chosenType == 1 )	{
			TrainHog th;
			if(std::stoi(parser.get<std::string>("verbose")) == 1)
				th.printSettings();
			th.train(false);
		}
		else if (chosenType == 2 ) {
			CombinedTrainHog cth;
			cth.train();
		}
		else if(chosenType == 3)  {
			TrainFHog tfh;
			tfh.train();
		}
		else
			std::cout << "Bad selection.\n";
	}
}

void mainFun::camera(cv::CommandLineParser parser)
{
	int typeAlg;
	
	std::cout << "\nSelect detection algorithm: \n 1) Only HoG (openCV) \n 2) MOG + HoG (openCV) \n";
	std::cout << " 3) only FHoG (dlib) \n 4) MOG + FHoG(dlib)  \n";
	std::cout << " 5) cascade classificator \n";
	std::cin >> typeAlg;
	if (typeAlg <= 0 || static_cast<unsigned>(typeAlg) > 6) {
				std::cout << "Bad selection.\n";
				return;
	}
	Pipeline pl;

	pl =  Pipeline(parser.get<std::string>("class"), typeAlg);
	std::cout << "camera" << std::endl;
	pl.execute(std::stoi(parser.get<std::string>("camera")));

}

//void mainFun::image(cv::CommandLineParser parser)
//{
//	int typeAlg;
//
//	std::cout << "\nSelect detection algorithm: \n 1) Only HoG (openCV) \n 2) only FHoG (dlib) \n";
//	std::cin >> typeAlg;
//	if (typeAlg == 1 || typeAlg == 2) {
//		if (typeAlg == 2) typeAlg = 3;
//
//		Pipeline *pl = new Pipeline(parser.get<std::string>("class"), algType);
//		Utils::setEvaluationFiles(parser.get<std::string>("image"));
//
//		pl->executeImages(parser.get<std::string>("image"));
//		std::map<std::string, int> maps;
//		pl->evaluate(maps);
//
//		delete pl;
//	}
//	else { std::cout << "Bad selection.\n"; }
//
//}

void mainFun::image(cv::CommandLineParser parser)
{
		while(true){
			Settings::getSettings("data/settings/settings_img.txt");
			Pipeline pl =  Pipeline(parser.get<std::string>("class"), 1);
			Utils::setEvaluationFiles(parser.get<std::string>("image"));

			pl.executeImages(parser.get<std::string>("image"));
			std::map<std::string, int> maps;
			pl.evaluate(maps);

			cv::waitKey(0);
	}

}

//void mainFun::video(cv::CommandLineParser parser)
//{
//	int typeAlg;
//
//	std::cout << "\nSelect detection algorithm: \n 1) Only HoG (openCV) \n 2) MOG + HoG (openCV) \n";
//	std::cout << " 3) only FHoG (dlib) \n 4) MOG + FHoG(dlib)  \n";
//	std::cout << " 7) TEST MODE \n" << std::endl;
//	std::cin >> typeAlg;
//
//	if (typeAlg == 0 || static_cast<unsigned>(typeAlg) > 6) {
//		std::cout << "Bad selection.\n";
//		return;
//	}
//	if( typeAlg == 7) { TestingPipeline("testingSVM.txt", "testingVideos.txt").execute(); return; }
//	Pipeline *pl = new Pipeline(parser.get<std::string>("class"), typeAlg);
//	Settings::nameFile = parser.get<std::string>("video");
//
//	replace(Settings::nameFile.begin(), Settings::nameFile.end(), '/', '-');
//	replace(Settings::nameFile.begin(), Settings::nameFile.end(), '.', '-');
//	Settings::nameTrainedFile = "data//trained//" + Settings::nameFile;
//	Settings::nameFile = "data//tested//" + Settings::nameFile;
//	Settings::nameTrainedFile.append("_trained.txt");
//	Settings::nameFile.append(".txt");
//
//	auto startTime = std::chrono::high_resolution_clock::now();
//	pl->execute(parser.get<std::string>("video"));
//	auto endTime = std::chrono::high_resolution_clock::now();
//	double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
//
//	printResults(time);
//	pl->evaluate();
//	cv::waitKey(0);
//
//	delete pl;
//}

void mainFun::video(cv::CommandLineParser parser)
{
	//Settings::getSettings("data/settings/settings_vga.txt");

//	TestingPipeline("testing/testing.txt").execute();
//	return;
	std::string videos[] = {"video/cctv4.avi", "video/cctv4.avi", "video/cctv4.mov" };

	for (auto vid : videos) {
		std::cout << "\t\t VIDEO " << vid << " ______________" << std::endl;
			while (true)
			{
			Settings::getSettings("data/settings/settings_avi_def.txt");
			Pipeline pl = Pipeline("default", 2);
			//Pipeline pl =  Pipeline("pedDet1_5.svm", 4);
			//Utils::setEvaluationFiles(parser.get<std::string>("video"));
			Utils::setEvaluationFiles(vid);
			//Settings::nameFile = vid;// parser.get<std::string>("video");

				auto startTime = std::chrono::high_resolution_clock::now();
			//pl->execute(parser.get<std::string>("video"));
			pl.execute(vid);
			auto endTime = std::chrono::high_resolution_clock::now();
			double time = std::chrono::duration<double, std::milli>(endTime - startTime ).count();

			printResults(time);
			std::map<std::string, int> maps;
			pl.evaluate(maps);
			cv::waitKey(0);			
		}
	}
		std::cout << "END";
}

void mainFun::extract(cv::CommandLineParser parser)
{
	int nRects;
	std::cout << "extracting ROI\n How many people are in stream? ( max 5)" << std::endl;
	std::cin >> nRects;
	if ( static_cast<unsigned>(nRects - 1) >= 5)
	{
		std::cout << "Bad chosen, 2 people selected." << std::endl;
		nRects = 2;
	}
	ExtractorROI eroi = ExtractorROI(nRects);
	eroi.extractROI(parser.get<std::string>("extract"));
}

void mainFun::createSample(cv::CommandLineParser parser)
{
	std::cout << "Creating samples from img...";
	clock_t timer = clock();
	Utils::createSamplesFromImage(parser.get<std::string>("createSample"), "makedSamples");
	timer = clock() - timer;
	std::cout << "DONE!\nParsing took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
}

void mainFun::printResults(clock_t timer)
{
	std::cout << "FPS: " << VideoStream::fps << "." << std::endl;
	std::cout << "ALG FPS: " << VideoStream::totalFrames / (static_cast<float>(timer) / CLOCKS_PER_SEC) << "." << std::endl;
	std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
	std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s."<< std::endl;
	std::cout << "Detection took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
}


///////////////////////
//		END			//
////////////////////////////////////////////////////////

////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

int Settings::mogHistory = 115;
double Settings::mogThresh = 8;
bool Settings::mogDetectShadows = true;
int Settings::cvxHullExtSize = 10;
int Settings::cvxHullExtTimes = 4;
double Settings::cvxHullThresh = 180;
double Settings::cvxHullMaxValue = 255;
cv::Size Settings::pedSize = cv::Size(48, 96);
int Settings::blockSize = 16;
int Settings::cellSize = 8;
int Settings::strideSize = 8;
int Settings::maxIterations = 1000;
int Settings::termCriteria = 3;
int Settings::kernel = 0;
int Settings::type = 100;
double Settings::epsilon = 1.e-06;
double Settings::coef0 = 0;
int Settings::degree = 3;
double Settings::gamma = 0.0025;
double Settings::paramNu = 0;
double Settings::paramP = 0;
double Settings::paramC = 0.06250;

std::string Settings::samplesPos = "samples/posSamples.txt";
std::string Settings::samplesNeg = "samples/negSamples.txt";
std::string Settings::samplesPosTest = "samples/listPosTestMin.txt";
std::string Settings::samplesNegTest = "samples/listNegTestMin.txt";

std::string Settings::classifierName2Train = "classifier";
int Settings::dilationSize = 3;
int Settings::erosionSize = 2;

cv::Size Settings::hogBlurFilter = cv::Size(6,6);
double Settings::hogHitTreshold = 0.92;
cv::Size Settings::hogWinStride = cv::Size(8, 8);
cv::Size Settings::hogPadding = cv::Size(0, 0);
double Settings::hogScale = 1.1;
double Settings::hogFinalTreshold = 0.95;
bool Settings::hogMeanshiftGrouping = 0;
int Settings::hogGroupTreshold = 2;
double Settings::hogEps = 0.8;
int Settings::hogMinArea = 4999;
cv::Size Settings::cropHogBlurFilter = cv::Size(0, 0);
double Settings::cropHogHitTreshold = 0.878;
cv::Size Settings::cropHogWinStride = cv::Size(4, 4);
cv::Size Settings::cropHogPadding = cv::Size(0, 0);
double Settings::cropHogScale = 1.09;
double Settings::cropHogFinalTreshold = 0;
bool Settings::cropHogMeanshiftGrouping = 0;
int Settings::cropHogGroupTreshold = 2;
double Settings::cropHogEps = 0.8;
int Settings::cropHogMinArea = 4999;
int Settings::fhogMinArea = 4999;
double Settings::fhogAdjustTreshold = 0.2222;
int Settings::cropFhogMinArea = 6000;
double Settings::cropFhogAdjustTreshold = 0;
