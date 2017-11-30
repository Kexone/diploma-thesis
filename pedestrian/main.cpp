#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include <opencv2/core/utility.hpp>
////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

std::string filename = "C:/Users/Jakub/Downloads/cctv4.mp4";
std::string posSamples = "listPos.txt";
std::string negSamples = "listNeg.txt";

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
		"{type  t        |detection | type of alg (train, test, video, picture}"
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
	else if (parser.has("train"))
	{
		std::cout << "training";
		train();
	}
	else if(parser.has("camera"))
	{
		std::cout << "camera";
		pl.execute(0);
	}
	else if(parser.has("video"))
	{
		clock_t timer;
		timer = clock();
		pl.execute(filename);
		timer = clock() - timer;
		printResults(timer);
		cv::waitKey(0);
	}
	
	return 0;
}

void train()
{
	TrainHog th;
	th.fillVectors(posSamples);
	th.fillVectors(negSamples, true);
	th.train();
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