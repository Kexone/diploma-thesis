#include <iostream>
#include <cstdlib>
#include <opencv2/core/utility.hpp>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include "source/train/combinedTrainHog.h"
#include "source/utils/extractorROI.h"
#include "source/utils/utils.h"
#include "source/test/testClass.h"
#include <csignal>
#include "source/train/trainCascade.h"
////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

std::string posSamples = "samples/posSamples1.txt";
std::string negSamples = "samples/negSamples1.txt";

///////////////////////
//					//
///////////////////////////////////////////////////////
//	 DECLARATION	 //
//////////////////////
namespace mainFun {
	void train(cv::CommandLineParser parser);
	void type(cv::CommandLineParser parser);
	void camera(cv::CommandLineParser parser);
	void video(cv::CommandLineParser parser);
	void image(cv::CommandLineParser parser);
	void extract(cv::CommandLineParser parser);
	void createSample(cv::CommandLineParser parser);
	/**
	* Print the results on screen
	*
	* @param timer represent time of the duration of the algorithm
	*
	*/
	void printResults(clock_t timer);
}


///////////////////////
//					//
///////////////////////////////////////////////////////
//		 MAIN		 //
//////////////////////

/* 
 * @TODO docs on trainfHog, hog, videostream, mediafile, utils, fhog, cascadeClass
 * @TODO add choose to set all params
 * @TODO ROC curves
 * 
 * @TODO calc confidence
 * @TODO train Dlib SVM from OpenCV HOG features
 * 
 * @TODO train cascade classificator
 * @TODO HAAR cascade classificator
 * @TODO LBP cascade classificator
 * @TODO ADA BOOST train
 * @TODO LBP train
 * @TODO HAAR train
 * 
 * @TODO refactor Utils class
 * @TODO implement cv::groupRectangles();
 * @TODO own implementation of detectMultiScale()

 */
std::string Settings::nameFile = "";
std::string Settings::nameTrainedFile = "";
bool Settings::showVideoFrames = false;

int main(int argc, char *argv[])
{
	if(argc < 2)
	{
		std::cout << "\tType -help for help" << std::endl;
		return 0;
	}
	const cv::String keys =
		"{ help h ?           |         |  print help message                       }"
		"{ alg	              |    1    |  alg type                                 }"
		"{ video v            |         |  use video as input                       }"
		"{ image i            |         |  use list of images as input              }"
		"{ camera c           |         |  enable camera capturing                  }"
		"{ class svm          | 48_96_16_8_8_9_01.yml |  trained clasifier path                   }"
		//"{ class svm          | default |  trained clasifier path                   }"
		"{ type  t            |         |  type of alg (train, test)                }"
		"{ extract e          |         |  extract ROI from videostream             }"
		"{ vizualize          |    1    |  show result in window                    }"
		"{ createSample cs    |    0    |  creating samples from image              }"
		;
	
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("DIPLOMA THESIS - Pedestrian Detection v1.0.0");
	if (parser.get<std::string>("vizualize") == "1" || parser.get<std::string>("vizualize") == "true") {
		Settings::showVideoFrames = true;
	}
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

		std::cout << "\n1) openCV SVM train \n2) combined train (extract features by opencv HOG and train by dlib SVM) \n \
		3) dlib SVM train \n4) cascade classificator train \nType of train: ";
		std::cin >> chosenType;

		if (chosenType == 1 )	{  // @TODO train from mat and select own params?
			//TrainHog th;
			TrainHog th = TrainHog(450, 3, 0, 100, 1.e-06, 0, 3, 0.0005, 0, 0, 0.0001, "2111_79_98.4.yml");
			//TrainHog th = TrainHog(114, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.313903, 0.212467, 0.130589, "2111_79_98.4.yml");
			//th.trainFromMat("test.yml", "labels.txt");
			th.train(posSamples, negSamples, false);
		}
		else if (chosenType == 2 ) {
			CombinedTrainHog cth;
			cth.train(posSamples, negSamples);
		}
		else if(chosenType == 3)  {
			TrainFHog tfh;
			tfh.train(posSamples, negSamples);
		}
		else if (chosenType == 4) {
			TrainCascade tc = TrainCascade("","","");
			tc.train();
		}
		else
			std::cout << "Bad selection.\n";
			return;
	}
}

void mainFun::camera(cv::CommandLineParser parser)
{
	Pipeline *pl;
	pl = new Pipeline(parser.get<std::string>("class"));
	std::cout << "camera" << std::endl;
	pl->execute(0);

	delete pl;
}

void mainFun::image(cv::CommandLineParser parser)
{
	Pipeline *pl = new Pipeline(parser.get<std::string>("class"));
	pl->executeImages(parser.get<std::string>("image"));
	std::cout << parser.get<std::string>("image") << std::endl;
	cv::waitKey(0);

	delete pl;
}

void mainFun::video(cv::CommandLineParser parser)
{
	Pipeline *pl = new Pipeline(parser.get<std::string>("class"));
	int typeAlg;
	clock_t timer;
	
	std::cout << "\nSelect detection algorithm: \n 1) Only HoG (openCV) \n 2) Mixtured HoG (openCV) \n 3) only FHoG (dlib) \n 4) mixtured FHoG (dlib)  \n 5) cascade classificator \n 6) TEST MODE \n" << std::endl;
	std::cin >> typeAlg;
	
	if(typeAlg == 0 || static_cast<unsigned>(typeAlg) > 6)	{
		std::cout << "Bad selection.\n";
		return;
	}

	timer = clock();
	Settings::nameFile = parser.get<std::string>("video");

	std::replace(Settings::nameFile.begin(), Settings::nameFile.end(), '/', '-');
	std::replace(Settings::nameFile.begin(), Settings::nameFile.end(), '.', '-');
	Settings::nameTrainedFile = "data//trained//" + Settings::nameFile;
	Settings::nameFile = "data//tested//" + Settings::nameFile;
	Settings::nameTrainedFile.append("_trained.txt");
	Settings::nameFile.append(".txt");

	pl->execute(parser.get<std::string>("video"),typeAlg);
	timer = clock() - timer;
	printResults(timer);
	pl->evaluate();
	cv::waitKey(0);

	delete pl;
}

void mainFun::extract(cv::CommandLineParser parser)
{
	int nRects;
	std::cout << "extracting ROI\n How many persons are in stream? ( max 5)" << std::endl;
	std::cin >> nRects;
	if ( static_cast<unsigned>(nRects - 1) >= 5)
	{
		std::cout << "Bad chosen. Selected 2 persons." << std::endl;
		nRects = 2;
	}
	ExtractorROI eroi = ExtractorROI(nRects);
	eroi.extractROI(parser.get<std::string>("extract"));
}

void mainFun::createSample(cv::CommandLineParser parser)
{
	std::cout << "Creating samples from img" << std::endl;
	clock_t timer = clock();
	Utils::createSamplesFromImage(parser.get<std::string>("createSample"), "makedSamples");
	timer = clock() - timer;
	std::cout << "Parsing took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
}

void mainFun::printResults(clock_t timer)
{
	std::cout << "FPS: " << VideoStream::fps << "." << std::endl;
	std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
	std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s."<< std::endl;
	std::cout << "Detection took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
	std::cout << "Possibly detection: " << Pipeline::allDetections << " frames." << std::endl;
}


///////////////////////
//		END			//
///////////////////////////////////////////////////////