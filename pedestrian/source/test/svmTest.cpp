/* 
 * 
 * Testing trained SVM  on samples data
 * 
 */

#include "svmTest.h"

 // The number of clock ticks per minute
#define CLOCKS_PER_MIN  ((clock_t)60000)

int counterTest = 1;
std::string classifierName = "test.yml";
std::string posTest = "samples/listPosTestMin.txt";
std::string negTest = "samples/listNegTestMin.txt";
std::vector< cv::Mat > posTestLst;
std::vector< cv::Mat > negTestLst;

SvmTest::SvmTest()
{
	this->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	this->kernel = cv::ml::SVM::LINEAR;
	this->type = cv::ml::SVM::NU_SVC;
	this->epsilon = 1.e-6;
	this->coef0 = 0.0;
	this->degree = 3;
	this->gamma = 0.1;
	if (posTestLst.empty() || negTestLst.empty()) {
		loadMats(posTest, posTestLst);
		loadMats(negTest, negTestLst);
	}
	posSamplesMin = "samples/listPosMin.txt";
	negSamplesMin = "samples/listNegMin.txt";


}

void SvmTest::setParams(int maxIter, double nu, double c, double p)
{
	this->maxIterations = maxIter;
	this->nu = nu;
	this->c = c;
	this->p = p;
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


float SvmTest::process(std::stringstream &ss)
{
	int valuation[] = { 0, 0, 0, 0 };	//nTruePos , nFalsePos, nTrueNeg, nFalseNeg


	std::cout << std::endl << counterTest << ". ITERATION OF TESTING" << std::endl;

	TrainHog th = TrainHog(maxIterations, termCriteria, kernel, type, epsilon, coef0, degree, gamma, nu, p, c, classifierName);

	trainTime = clock();
	th.fillVectors(posSamplesMin);
	th.fillVectors(negSamplesMin, true);
	th.train(false);
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
	print2File(counterTest, valuation, ss);
	std::cout << "ACCURACY " << accuracy << "%"<< std::endl;

	counterTest++;
	return accuracy;
}

void SvmTest::print2File(int currentTestNumb,int *valuation, std::stringstream &ss)
{
	std::ofstream file;
	file.open("result.txt", std::ios::app);

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
	file << ss.str() << std::endl;
	file << "\tTRAIN TIME: " << static_cast<float>(trainTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl;
	file << "\t__SVM RESULTS__" << std::endl;
	file << "POS GOOD: " << valuation[0] << " POS BAD: " << valuation[1] << std::endl;
	file << "NEG GOOD: " << valuation[2] << " NEG BAD: " << valuation[3] << std::endl;
	file << "ACCURACY: " << accuracy << " %" << std::endl;
	file << "CLASS TIME: " << static_cast<float>(classTime / CLOCKS_PER_MIN) << " MIN" << std::endl << std::endl;
	file << "\t<< END" << currentTestNumb << ".ITERATION>>" << std::endl;

	file.close();
	ss.str("");
	ss.clear();
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