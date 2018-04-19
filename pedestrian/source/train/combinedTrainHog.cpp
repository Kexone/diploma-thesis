#include "stdafx.h"
#include "combinedTrainHog.h"


void CombinedTrainHog::train()
{
//	std::vector< cv::Mat > posSamplesLst;
//	std::vector< cv::Mat > negSamplesLst;
//	std::vector< cv::Mat > gradientLst;
	std::vector< int > labels;
//	cv::Size pedestrianSize = Settings::pedSize;
	cv::Mat trainMat;

//	Utils::fillSamples2List(Settings::samplesPos, posSamplesLst, pedestrianSize);

//	labels.assign(posSamplesLst.size(), +1);
//	const unsigned int old = static_cast<unsigned int>(labels.size());

//	Utils::fillSamples2List(Settings::samplesNeg, negSamplesLst, pedestrianSize);

//	labels.insert(labels.end(), negSamplesLst.size(), -1);
//	CV_Assert(old < labels.size());

	//std::cout << "Positive samples: " << posSamplesLst.size() << std::endl;
	//std::cout << "Negative samples: " << negSamplesLst.size() << std::endl;

	//extractFeatures(posSamplesLst, gradientLst);
	//extractFeatures(negSamplesLst, gradientLst);
	calcMatForTraining(trainMat, labels, true);
	
	TrainFHog::train(trainMat, labels);
}
