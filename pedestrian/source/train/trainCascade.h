#ifndef TRAIN_CASCADE_H
#define TRAIN_CASCADE_H

#include <opencv2/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/objdetect/objdetect_c.h>
//#include "../../3dparty/openCV/cascadeClassifier.h"
//#include "../../3dparty/openCV/lbpfeatures.h"
//#include "../../3dparty/openCV/HOGfeatures.h"
class TrainCascade
{
private:
	
	//CvCascadeClassifier _classifier;
	//CvCascadeParams _cascadeParams;
	//CvCascadeBoostParams _stageParams;
	//std::string _classifierPath, _posSamples, _negSamples;
	//int _numPos = 2000;
	//int _numNeg = 1000;
	//int _numStages = 20;
	//int _numThreads = cv::getNumThreads();
	//int _precalcValBufSize = 1024,
	//	_precalcIdxBufSize = 1024;
	//bool _baseFormatSave = false;
	//double _acceptanceRatioBreakValue = -1.0;

public:
	TrainCascade(std::string classPath, std::string posSamples, std::string negSamples);
	void train();
};

#endif //TRAIN_CASCADE_H