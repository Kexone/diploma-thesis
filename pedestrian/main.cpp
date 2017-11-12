#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"

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
	Pipeline pl;
	if( argc > 1)
	{
		if (std::strcmp(argv[1], "train") == 0)
		{
			std::cout << "training";
			train();
		}
		else if(std::strcmp(argv[1], "camera") == 0)
		{
			std::cout << "camera";
			pl.execute(0);
		}
		else
		{
			std::cout << "Bad params";
		}
	}
	else
	{
		clock_t timer;
		timer = clock();
		pl.execute(filename);
		timer = clock() - timer;
		printResults(timer);
	}
	
	cv::waitKey(0);
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
	std::cout << VideoStream::fps << std::endl;
	std::cout << VideoStream::totalFrames << std::endl;
	std::cout << VideoStream::fps / static_cast<float>(VideoStream::totalFrames) << std::endl;
	std::cout << "Detection took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
}


///////////////////////
//		END			//
///////////////////////////////////////////////////////