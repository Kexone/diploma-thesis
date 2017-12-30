#ifndef COMBINEDTRAINHOG_H
#define COMBINEDTRAINHOG_H
#include <string>
#include "trainhog.h"
#include "trainfhog.h"

/**
 * class CombinedTrainHog
 * 
 * This class combining calculated features from OpenCV HOG to Dlib SVM classification training
 */
class CombinedTrainHog : public TrainHog, TrainFHog
{

public:

	/**
	* @brief This method combine OpenCV computing HOG features and dlib training
	* Output from this method is Dlib trained classifier
	*
	* @param posSamplesPath path of positive samples
	* @param negSamplesPath path of negative samples
	*/
	void train(std::string posSamplesPath, std::string negSamplesPath);
	
};


#endif // COMBINEDTRAINHOG_H