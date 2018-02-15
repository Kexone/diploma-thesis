#include "trainCascade.h"
#include <iostream>
#include <sstream>


//@TODO make choosable params
TrainCascade::TrainCascade()
{
	 
}

void TrainCascade::execute(bool creatingSamples)
{
#if _WIN64 && NDEBUG
#define ENVIRONMENT64
	int type;
	if (creatingSamples)
		createSamples();

	std::cout << "Would you like to use default params? (1/0)" << std::endl;
	std::cin >> type;
	if (type == 1)
		train();
#else
#define ENVIRONMENT32
	std::cout << "TRAINING CASCADES IS AVAILABLE ONLY UNDER WINDOWS 64bit" << std::endl;

#endif

}



void TrainCascade::train()
{
	std::stringstream ss;
	ss << "opencv_traincascade.exe -data data -vec " << _vec << " -bg " << _bg << " -numPos " << _numPos << " -numNeg " << _numNeg << " -numStages ";
	ss << _numStages << " -numThreads " << _numStages << " -stageType " << _stageType << " -featureType " << _featureType;
	ss << " -w " << _width << " -h " << _height << " -minHitRate " << _minHitRate << " -maxFalseAlarmRate " << _maxFalseAlarmRate;
	ss << " -maxDepth " << _maxDepth << " -maxWeakCount " << _maxWeakCount;
	system(ss.str().c_str());
	ss.str("");
	ss.clear();
}

void TrainCascade::createSamples()
{
	std::stringstream ss;
	ss << "opencv_createsamples.exe -vec " << _vec << " -bg " << _bg << " -num " << _num << " -img " << _image;
	ss << " -w " << _width << " -h " << _height << "-maxxangle " << _maxxAngle << " -maxyangle " << _maxyAngle << " -maxzangle ";
	ss << _maxyAngle << " -maxidev " << _maxIdev << "-bgcolor " << _bgColor << " -bgthresh " << _bgThresh;
	system(ss.str().c_str());
	ss.str("");
	ss.clear();
}