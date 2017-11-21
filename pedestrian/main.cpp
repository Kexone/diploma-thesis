#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include "source/test/svmTest.h"
#include "3dparty/de/DifferentialEvolution.h"
#include "3dparty/de/TestFunctions.h"

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
	if( argc > 1)
	{
		if (std::strcmp(argv[1], "train") == 0)
		{
			std::cout << "training" << std::endl;
			train();
		}
		else if (std::strcmp(argv[1], "camera") == 0)
		{
			Pipeline pl;
			std::cout << "camera" << std::endl;
			pl.execute(1);
		}
		else if (std::strcmp(argv[1], "testsvm") == 0)
		{
			SvmTest svm;
			svm.runSvmTest();
		}
		else
		{
			std::cout << "Bad params" << std::endl;
		}
	}
	else
	{
		Pipeline pl;
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
	//th.fillVectors(posSamples);
	//th.fillVectors(negSamples, true);
	//th.train(false);
	th.trainFromMat("test.yml", "labels.txt");
}

void printResults(clock_t timer)
{
	std::cout << "FPS: " << VideoStream::fps << "." << std::endl;
	std::cout << "Algorithm FPS: " << VideoStream::totalFrames / (static_cast<float>(timer) / CLOCKS_PER_SEC) << "." << std::endl;
	std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
	std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s."<< std::endl;
	std::cout << "Detection took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
	std::cout << "Possibly detection: " << Pipeline::allDetections << " frames." << std::endl;
}


///////////////////////
//		END			//
///////////////////////////////////////////////////////