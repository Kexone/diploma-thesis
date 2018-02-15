#include "combinedTrainHog.h"
#include "../utils/utils.h"

void CombinedTrainHog::train(std::string posSamplesPath, std::string negSamplesPath)
{
	std::vector< cv::Mat > posSamplesLst;
	std::vector< cv::Mat > negSamplesLst;
	std::vector< cv::Mat > gradientLst;
	std::vector< int > labels;
	cv::Size pedestrianSize = getPedSize();
	cv::Mat trainMat;

	Utils::fillSamples2List(posSamplesPath, posSamplesLst, labels, pedestrianSize, false, true);
	Utils::fillSamples2List(negSamplesPath, negSamplesLst, labels, pedestrianSize, true, true);

	std::cout << "Positive samples: " << posSamplesLst.size() << std::endl;
	std::cout << "Negative samples: " << negSamplesLst.size() << std::endl;

	extractFeatures(posSamplesLst, gradientLst);
	extractFeatures(negSamplesLst, gradientLst);
	
	TrainFHog::train(gradientLst, labels);
}
