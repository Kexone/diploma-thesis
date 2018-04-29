#ifndef COMBINEDTRAINHOG_H
#define COMBINEDTRAINHOG_H
#include <string>
#include "trainhog.h"
#include "trainfhog.h"
#include "../utils/utils.h"
#include "../settings.h"
/**
 * @class CombinedTrainHog
 * 
 *@brief This class combining calculated features from OpenCV HOG to Dlib SVM classification training
 */
class CombinedTrainHog : public TrainHog, TrainFHog
{

public:

	/**
	* @brief This method combine OpenCV computing HOG features and dlib training
	* Output from this method is Dlib trained classifier
	*
	*/
	void train();
	
};


#endif // COMBINEDTRAINHOG_H