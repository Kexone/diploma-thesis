#include <iostream>
#include "source/train/trainhog.h"
#include "source/pipeline.h"
#include <opencv2/core/utility.hpp>
#include "source/train/combinedTrainHog.h"
#include "source/utils/extractorROI.h"
#include "source/utils/utils.h"
#include "source/test/svmTest.h"
#include "source/test/testClass.h"

////////////////////////////////////////////////////////
//		DATA		 //
//////////////////////

std::string filename = "C:/Users/Jakub/Downloads/cctv2.mp4";
std::string posSamples = "samples/posSamples.txt";
std::string negSamples = "samples/negSamples.txt";
std::string posSamplesMin = "samples/listPosMin.txt";
std::string negSamplesMin = "samples/listNegMin.txt";

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
 * @TODO docs
 * @TODO add choose to set all params
 * @TODO ROC curves
 * 
 * 
 * @TODO calc confidence
 * @TODO testing cycl for c_svc type --DONE
 * @TODO clear bad samples from dataset
 * @TODO train on same samples --DONE
 * @TODO saving ROI frames from HOG --DONE
 * @TODO train on siluette samples
 * 
 * @TODO train cascade classificator
 * @TODO HAAR cascade classificator
 * @TODO LBP cascade classificator
 * @TODO ADA BOOST train
 * @TODO LBP train
 * @TODO HAAR train
 * @TODO replace convex hull with something more effiness
 * 
 * 
 * @TODO optimalize pipeline for all algorithms
 * @TODO refactor Utils class
 * @TODO implement cv::groupRectangles();
 * @TODO own implementation of detectMultiScale()
 * @TODO method for cropping from img with sliding window
 * @TODO sliding window for negative samples
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
		"{extract e      |          | extract ROI from videostream}"
		"{vizualize      | 0        | show result in window   }"
		"{cs createSample      | 0        |  creating samples from image  }"
		;

	cv::CommandLineParser parser(argc, argv, keys);
	parser.about("DIPLOMA THESIS -- Pedestrian Detection v1.0.0");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	else if (parser.has("type"))	{
		std::cout << "training" << std::endl;
		std::string type = parser.get<std::string>("type");
		if(!type.compare("test") )
		{
			TestClass tc;
			tc.initTesting();
		}
		std::cout << parser.get<std::string>("type") << std::endl;
		if (!type.compare("train"))
		{
			train();
		}
		//TestClass ts = TestClass();
	//	ts.initTesting();
		//CombinedTrainHog cth;
		//cth.train(posSamplesMin, negSamplesMin);
		//TrainFHog tfh;
		//tfh.train(posSamples,negSamples);
		//return 0;
	}
	else if (parser.has("camera"))	{
		Pipeline pl;
		std::cout << "camera" << std::endl;
		pl.execute(0);
	}
	else if (parser.has("video"))	{
		Pipeline pl;
		clock_t timer;
		timer = clock();
		pl.execute(parser.get<std::string>("video"));
		timer = clock() - timer;
		printResults(timer);
		pl.evaluate("test.txt", "Result.txt");
		cv::waitKey(0);
	}
	else if (parser.has("image"))	{
		Pipeline pl;
		pl.executeImages(parser.get<std::string>("image"));
		std::cout << parser.get<std::string>("image") << std::endl;
		cv::waitKey(0);
	}

	else if (parser.has("extract")) {
		std::cout << "extracting ROI" << std::endl;
		ExtractorROI eroi = ExtractorROI(2);
		eroi.extractROI(parser.get<std::string>("extract"));
	}

	else if (parser.has("createSample")) {
		std::cout << "Creating samples from img" << std::endl;
		clock_t timer = clock();
		Utils::createSamplesFromImage(parser.get<std::string>("createSample"), "makedSamples");
		timer = clock() - timer;
		std::cout << "Parsing took " << static_cast<float>(timer) / CLOCKS_PER_SEC << "s." << std::endl;
	}
	
	return 0;
}

void train()
{
	TrainHog th;
	//TrainHog th = TrainHog(114, 3, 0, 100, 1.e-06, 0, 3, 0.1, 0.313903, 0.212467, 0.130589, "2111_79_98.4.yml");
	th.train("samples/silhouettesPos.txt", negSamples, false);
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