#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include <opencv2/core/utility.hpp>
#include "source/train/combinedTrainHog.h"
#include "source/utils/extractorROI.h"

////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

std::string filename = "C:/Users/Jakub/Downloads/cctv2.mp4";
std::string posSamples = "samples/listPos.txt";
std::string negSamples = "samples/listNeg.txt";
std::string posSamplesMin = "samples/listPosMinMin.txt";
std::string negSamplesMin = "samples/listNegMinMin.txt";

///////////////////////
//					//
///////////////////////////////////////////////////////
//	 DECLARATION	 //
//////////////////////

void train();
void printResults(clock_t timer);

///////////////////////
//					//
///////////////////////////////////////////////////////
//		 MAIN		 //
//////////////////////

/*
 * @TODO command line parser
 * @TODO train cascade classificator
 * @TODO HAAR cascade classificator
 * @TODO LBP cascade classificator
 * @TODO ADA BOOST train
 * @TODO LBP train
 * @TODO HAAR train
 * @TODO optimalize pipeline for all algorithms
 * @TODO replace convex hull with something more effiness
 * @TODO refactor Utils class
 * @TODO docs
 */
int main(int argc, char *argv[])
{
	const cv::String keys =
		"{help h ? || print help message}"
		"{alg	         |1         | alg type}"
		"{video v        |          | use video as input}"
		"{image i        |          | use list of images as input}"
		"{camera c       |          | enable camera capturing}"
		"{class svm      |0         | trained clasifier path }"
		"{type  t        |          | type of alg (train, test, video, picture}"
		"{vizualize      | 0        | show result in window   }"
		;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("DIPLOMA THESIS -- Pedestrian Detection v1.0.0");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	else if (parser.has("type"))
	{
		std::cout << "training" << std::endl;
		std::cout << parser.get<std::string>("type") << std::endl;
		//train();
		CombinedTrainHog cth;
		cth.train(posSamplesMin, negSamplesMin);
	}
	else if (parser.has("camera"))
	//TrainFHog tfh;
	//tfh.train(posSamples,negSamples);
	//return 0;
	if( argc > 1)
	{
		Pipeline pl;
		std::cout << "camera";
		ExtractorROI eroi = ExtractorROI(3,"Result.txt");
		eroi.extractROI(parser.get<std::string>("camera"));
		pl.execute(0);
	}
	else if (parser.has("video"))
	{
		Pipeline pl;
		clock_t timer;
		timer = clock();
		pl.execute(parser.get<std::string>("video"));
		timer = clock() - timer;
		printResults(timer);
		cv::waitKey(0);
	}
	else if (parser.has("i"))
	{
		Pipeline pl;
		clock_t timer;
		timer = clock();
		pl.executeImages(parser.get<std::string>("image"));
		std::cout << parser.get<std::string>("image");
		timer = clock() - timer;
		//printResults(timer);
		cv::waitKey(0);
	}
	
	return 0;
}

void train()
{
	TrainHog th = TrainHog(114, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.313903, 0.212467, 0.130589, "2111_79_98.4.yml");
	th.train(posSamples, negSamples, false);
	//th.trainFromMat("test.yml", "labels.txt");
}

void printResults(clock_t timer)
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