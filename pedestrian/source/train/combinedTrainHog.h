#ifndef COMBINEDTRAINHOG_H
#define COMBINEDTRAINHOG_H
#include <string>
#include "trainhog.h"
#include "trainfhog.h"

/**
 * class CombinedTrainHog
 * 
 * This class combining calculated features from OpenCV HOG to Dlib SVM classification training
 * @TODO train Dlib SVM from OpenCV HOG features
 */
class CombinedTrainHog : public TrainHog, TrainFHog
{

public:

	/**
	* @brief
	*
	* @param
	*/
	void train(std::string posSamplesPath, std::string negSamplesPath);
	
};


#endif // COMBINEDTRAINHOG_H