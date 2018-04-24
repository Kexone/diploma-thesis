#include "combinedTrainHog.h"


void CombinedTrainHog::train()
{
	std::vector< int > labels;
	cv::Mat trainMat;

	calcMatForTraining(trainMat, labels, true);
	TrainFHog::train(trainMat, labels);
}
