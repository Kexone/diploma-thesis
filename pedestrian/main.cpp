#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include <opencv2/core/utility.hpp>
////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

std::string filename = "C:/Users/Jakub/Downloads/cctv4.mp4";
std::string posSamples = "samples/listPos.txt";
std::string negSamples = "samples/listNeg.txt";

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

	Pipeline pl;
	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("DIPLOMA THESIS- Pedestrian Detection v1.0.0");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	else if (parser.has("type"))
	{
		std::cout << "training";
		train();
	}
	else if (parser.has("camera"))
	{
		std::cout << "camera";
		pl.execute(0);
	}
	else if (parser.has("video"))
	{
		clock_t timer;
		timer = clock();
		pl.execute(parser.get<std::string>("video"));
		timer = clock() - timer;
		printResults(timer);
		cv::waitKey(0);
	}
	else if (parser.has("i"))
	{
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

	th = TrainHog(105, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.173182, 0.393894, 0.111534, "2292_78_98.3.yml");
	th.train(posSamples, negSamples, false);

	th = TrainHog(100, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.477977, 0.3514, 0.108495, "2717_78_98.4.yml");
	th.train(posSamples, negSamples, false);

	th = TrainHog(100, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.477977, 0.3514, 0.108495, "2717_78_98.4.yml");
	th.train(posSamples, negSamples, false);

	th = TrainHog(114, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.243877, 0.336372, 0.130589, "3111_79_98.4.yml");
	th.train(posSamples, negSamples, false);

	th = TrainHog(114, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.243877, 0.336372, 0.130589, "3111_79_98.4.yml");
	th.train(posSamples, negSamples, false);
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