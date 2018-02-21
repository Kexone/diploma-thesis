#include "combinedTrainHog.h"


void CombinedTrainHog::train()
{
	std::vector< cv::Mat > posSamplesLst;
	std::vector< cv::Mat > negSamplesLst;
	std::vector< cv::Mat > gradientLst;
	std::vector< int > labels;
	cv::Size pedestrianSize = Settings::pedSize;
	cv::Mat trainMat;

	Utils::fillSamples2List(Settings::samplesPos, posSamplesLst, labels, pedestrianSize, false, true);
	Utils::fillSamples2List(Settings::samplesNeg, negSamplesLst, labels, pedestrianSize, true, true);

	std::cout << "Positive samples: " << posSamplesLst.size() << std::endl;
	std::cout << "Negative samples: " << negSamplesLst.size() << std::endl;

	extractFeatures(posSamplesLst, gradientLst);
	extractFeatures(negSamplesLst, gradientLst);
	
	TrainFHog::train(gradientLst, labels);
}
