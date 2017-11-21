#ifndef HOG_H
#define HOG_H

#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <opencv2/videostab/ring_buffer.hpp>
#include <opencv2/highgui.hpp>
#include "../media/croppedimage.h"

class Hog
{
public:
    Hog();
	Hog(std::string svmPath);
    std::vector< std::vector < cv::Rect > > detect(std::vector< CroppedImage > &frame);
	void detect(std::vector< cv::Mat > &testLst, int &nTrue, int &nFalse, bool pedestrian = true);

private:
    void getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector );
    cv::HOGDescriptor hog;
	cv::Ptr<cv::ml::SVM> svm;
};

#endif // HOG_H
