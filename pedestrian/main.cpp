#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include "source/test/testClass.h"
#include "source/train/trainfhog.h"

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
	TrainFHog tfh;
	tfh.train(posSamples,negSamples);
	return 0;
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
			TestClass().initTesting();
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
	TrainHog th = TrainHog(114,3,0,100,1.e-06,0,3,0.1, 0.313903, 0.212467, 0.130589,"2111_79_98.4.yml");
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
	std::cout << "Algorithm FPS: " << VideoStream::totalFrames / (static_cast<float>(timer) / CLOCKS_PER_SEC) << "." << std::endl;
	std::cout << "Total frames: " << VideoStream::totalFrames << "." << std::endl;
	std::cout << "Video duration: " << VideoStream::totalFrames / static_cast<float>(VideoStream::fps) << "s."<< std::endl;
	std::cout << "Detection took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
	std::cout << "Possibly detection: " << Pipeline::allDetections << " frames." << std::endl;
}


///////////////////////
//		END			//
///////////////////////////////////////////////////////