/* 
 * 
 * Testing trained SVM  on samples data
 * 
 */

#include "svmTest.h"
#include "../utils/utils.h"

// The number of clock ticks per minute
#define CLOCKS_PER_MIN  ((clock_t)60000)

int counterTest = 1;


SvmTest::SvmTest()
{
	this->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	this->kernel = cv::ml::SVM::LINEAR;
	this->type = cv::ml::SVM::C_SVC;
	this->epsilon = 1.e-6;
	this->coef0 = 0.0;
	this->degree = 3;
	this->gamma = 0.1;
	this->classifierName = "test.yml";
	


}

void SvmTest::setParams(int maxIter, double nu, double c, double p)
{
	this->maxIterations = maxIter;
	this->nu = nu;
	this->c = c;
	this->p = p;
}

void SvmTest::setParams(int maxIter, double c, double gamma)
{
	this->maxIterations = maxIter;
	this->c = c;
	this->gamma = gamma;
}

void SvmTest::preprocessing()
{
	std::string posTest = "samples/posSamples.txt";
	std::string negTest = "samples/negSamples.txt";
	TrainHog th;
	posSamplesMin = "samples/listPos.txt";
	//posTest = "samples/listPosGridIlid.txt";
	//posSamplesMin = "samples/listPosTownCenterSarc3d.txt";
	//posTest = "samples/listPosTownCenterSarc3d.txt";
	//posSamplesMin = "samples/listPos3DPES_CAVIAR4REID.txt";
	negSamplesMin = "samples/negSamplesCustom.txt";
	th.calcMatForTraining(posSamplesMin, negSamplesMin,trainMat,labels);
	if (posTestLst.empty() || negTestLst.empty()) {
		loadMats(posTest, posTestLst);
		loadMats(negTest, negTestLst);
	}
}

void SvmTest::initResultFile(std::stringstream &ss)
{
	std::ofstream file;
	file.open("result.txt", std::ios::app);

	file << ss.str();
	file.close();
	ss.str("");
	ss.clear();
}


float SvmTest::process()
{
	int valuation[] = { 0, 0, 0, 0 };	//nTruePos , nFalsePos, nTrueNeg, nFalseNeg


	std::cout << std::endl << counterTest << ". ITERATION OF TESTING" << std::endl;

	TrainHog th = TrainHog(maxIterations, termCriteria, kernel, type, epsilon, coef0, degree, gamma, nu, p, c, classifierName);

	trainTime = clock();
	th.trainFromMat(trainMat, labels);
	trainTime = clock() - trainTime;

	Hog h = Hog(classifierName);

	std::cout << " << TESTING SVM >>" << std::endl;

	classTime = clock();
	h.detect(posTestLst, valuation[0], valuation[1]);
	h.detect(negTestLst, valuation[2], valuation[3], false);
	classTime = clock() - classTime;
	std::cout << "POS DETECTION [T/F] " << valuation[0]  << "/" << valuation[1] << std::endl;
	std::cout << "NEG DETECTION [T/F] " << valuation[2] << "/" << valuation[3] << std::endl;

	accuracy = static_cast<float>(valuation[0]  + valuation[2] ) / static_cast<float>(valuation[0]+ valuation[1]+ valuation[2]+ valuation[3]);
	print2File(counterTest, valuation);
	std::cout << "ACCURACY " << accuracy << " %"<< std::endl;

	counterTest++;
	return accuracy;
}

void SvmTest::print2File(int currentTestNumb,int *valuation)
{
	std::ofstream file;
	file.open("testinResults.txt", std::ios::app);

	file << "\n\t<< START" << currentTestNumb << ".ITERATION>>" << std::endl << std::endl;
	file << "\t__SVM SETTINGS__" << std::endl;
	file << "\tMAX ITER: " << this->maxIterations << std::endl;
	file << "\tTERM CRIT: " << this->termCriteria << std::endl;
	file << "\tKERNEL: " << this->kernel << std::endl;
	file << "\tTYPE SVM: " << this->type << std::endl;
	file << "\tEPSILON: " << this->epsilon << std::endl;
	file << "\tCOEF0:" << this->coef0 << std::endl;
	file << "\tDEGREE: " << this->degree << std::endl;
	file << "\tGAMMA: " << this->gamma << std::endl;
	file << "<< TESTED PARAMETERS >> " << std::endl;
	file << "\tNU: " << this->nu << std::endl;
	file << "\tP: " << this->p << std::endl;
	file << "\tC: " << this->c << std::endl;
	file << "\tTRAIN TIME: " << static_cast<float>(trainTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl;
	file << "\t__SVM RESULTS__" << std::endl;
	file << "POS GOOD: " << valuation[0] << " POS BAD: " << valuation[1] << std::endl;
	file << "NEG GOOD: " << valuation[2] << " NEG BAD: " << valuation[3] << std::endl;
	file << "ACCURACY: " << accuracy << " %" << std::endl;
	file << "CLASS TIME: " << static_cast<float>(classTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl;
	file << "\t<< END" << currentTestNumb << ".ITERATION>>" << std::endl;

	file.close();
	if((float)1 - ((float)valuation[1] / (float)valuation[0]) > 0.9888)
	{		
		std::cout << (float)1 - ((float)valuation[1] / (float)valuation[0])  << "  !!!!!nice in " << currentTestNumb << " iter/" << std::endl;
		std::cout << '\a';
	}
}


void SvmTest::loadMats(std::string samplesPath, std::vector< cv::Mat > &lst)
{
	assert(!samplesPath.empty());
	cv::Mat frame;
	lst.clear();
	std::fstream sampleFile(samplesPath);
	std::string oSample;
	while (sampleFile >> oSample) {

		frame = cv::imread(oSample, CV_32FC3);
		cv::resize(frame, frame, cv::Size(48, 96));
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		lst.push_back(frame.clone());
	}
}