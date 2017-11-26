#ifndef CASCADECLASS_H
#define CASCADECLASS_H

#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#include <opencv2/highgui.hpp>
#include "../media/croppedimage.h"

class CascadeClass
{
public:
	CascadeClass();
	std::vector< std::vector < cv::Rect > > detect(std::vector< CroppedImage > &frames);

private:
	cv::CascadeClassifier clasifier;
};

#endif //CASCADECLASS_H