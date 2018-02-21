#include "mog.h"
#include "../settings.h"
#include <opencv2/video/background_segm.hpp>

Mog::Mog()
{ 
   // pMOG1 = cv::bgsegm::createBackgroundSubtractorMOG(150,13,0.1,1); //MOG approach 
	// F1 score : 0.866755 - 20 , 0.870712 - 18, 0.871795 -17, 0.873276 - 16, 0.874754 -15, 0.876302 - 13,  0.883301 - 10, 0.88746 - 8,  0.885477 - 7
    pMOG2 = cv::createBackgroundSubtractorMOG2(Settings::mogHistory, Settings::mogThresh, Settings::mogDetectShadows); //MOG2 approach /*Settings::mogHistory, Settings::mogThreshold*/
} 
// F1 score : 0.88533 115.8.true
void Mog::processMat(cv::Mat &frame)
{
    pMOG2->apply(frame, frame);
   // pMOG1->apply(frame, frame);
}
