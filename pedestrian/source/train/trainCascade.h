#ifndef TRAIN_CASCADE_H
#define TRAIN_CASCADE_H
#include <string>

class TrainCascade
{
private:
	int _numPos = 1;
	int _numNeg = 300;
	int _numStages = 13;
	int _numThreads = 4;
	int _width = 16;
	int _height = 36;
	int _maxDepth = 1;
	int _maxWeakCount = 100;
	
	std::string _stageType = "BOOST";
	std::string _featureType = "LBP";
	std::string _vec = "pedestrian.vec";
	std::string _bg = "bg.txt";

	float _minHitRate = 0.955f;
	float _maxFalseAlarmRate = 0.42f;

	float _maxxAngle = 0.6f;
	float _maxyAngle = 0;
	float _maxzAngle = 0.3f;

	std::string _image = "0001_001.jpg";

	int _num = 100;
	int _maxIdev = 100;
	int _bgColor = 0;
	int _bgThresh = 0;

	void train();
	void createSamples();

public:
	TrainCascade();
	void execute(bool creatingSamples = true);
};

#endif //TRAIN_CASCADE_H