#ifndef TRAIN_CASCADE_H
#define TRAIN_CASCADE_H
#include <string>

class TrainCascade
{
private:
	


public:
	TrainCascade(std::string classPath, std::string posSamples, std::string negSamples);
	void train();
};

#endif //TRAIN_CASCADE_H