#include "trainhog.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
void TrainHog::fillVectors(std::string &path, bool isNeg)
{
    assert(!path.empty());
    std::vector<cv::Mat> data;
    int counter =0;
    cv::Mat frame;
	std::fstream sampleFile(path);
	std::string oSample;
	while (sampleFile >> oSample) {

		frame = cv::imread(oSample, CV_32FC3);
		if (frame.empty()) std::cout << "fail" << std::endl;
		cv::resize(frame, frame, pedestrianSize);
		data.push_back(frame.clone());
		if (!isNeg) {
			labels.push_back(1);
		}
		else {
			labels.push_back(0);
		}
		counter++;
	}
    if(!isNeg)
        posSamples = data;
    else
        negSamples = data;
    data.clear();
	std::string type = "Negative ";
	if (!isNeg)
		type = "Positive ";
	std::cout << type << "samples: " << counter << std::endl;

}

void TrainHog::trainFromMat(std::string matPath, std::string labelsPath)
{
	cv::Mat trainMat;
	cv::FileStorage fs(matPath, cv::FileStorage::READ);
	fs["samples"] >> trainMat;

	std::ifstream is(labelsPath);
	std::istream_iterator<int> start(is), end;
	labels = std::vector<int> (start, end);

	trainSvm(trainMat, labels);
}

void TrainHog::train(bool saveData)
{
    std::vector< cv::Mat > gradientLst;
    cv::Mat trainMat;
    extractFeatures(posSamples, gradientLst);
	posSamples.clear();
    extractFeatures(negSamples, gradientLst);
	negSamples.clear();
    convertSamples2Mat(gradientLst, trainMat);
	gradientLst.clear();

	if (saveData) saveLabeledMat(trainMat);

    trainSvm(trainMat, labels);
}

void TrainHog::extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst)
{
    cv::HOGDescriptor hog(
					pedestrianSize, //winSize
                    cv::Size(blockSize,blockSize), //blocksize
                    cv::Size(strideSize,strideSize), //blockStride,
                    cv::Size(cellSize,cellSize), //cellSize,
                    9, //nbins,
                    0, //derivAper,
                    -1, //winSigma,
                    0, //histogramNormType,
                    0.2, //L2HysThresh,
                    0 //gammal corRection,
                                    //nlevels=64
                    );
    cv::Mat gr;
    std::vector< cv::Point > location;
    std::vector< float > descriptors;
    for(auto &mat : samplesLst) {
        cv::cvtColor(mat, gr,cv::COLOR_BGR2GRAY);
        hog.compute(gr, descriptors,cv::Size(8, 8),cv::Size(0, 0),location);
        gradientLst.push_back( cv::Mat( descriptors ).clone() );
    }
}

void TrainHog::trainSvm(cv::Mat &trainMat, const std::vector<int> &labels)
{
    std::cout << "START training ..." << std::endl;
	clock_t timer = clock();

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

	svm->setCoef0(coef0);
	svm->setDegree(degree);
	svm->setTermCriteria(cv::TermCriteria(termCriteria, maxIterations, epsilon));
	svm->setGamma(gamma);
	svm->setKernel(kernel);//linear
	svm->setNu(nu);
	svm->setP(p);
	svm->setC(c);
	svm->setType(type);
	svm->train(trainMat, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	svm->save(classifierName);

	timer = clock() - timer;
    std::cout << "training DONE ..."<< static_cast<float>(timer) / (CLOCKS_PER_SEC*60) << " min" <<  std::endl;

}

void TrainHog::convertSamples2Mat(const std::vector<cv::Mat> &trainSamples, cv::Mat &trainData)
{
    const int rows = trainSamples.size();
    const int cols = std::max( trainSamples[0].cols, trainSamples[0].rows);
    cv::Mat tmp(1,cols,CV_32FC1);
    trainData = cv::Mat(rows,cols, CV_32FC1);
    int i = 0;
    for(auto sample : trainSamples) {
        assert(sample.cols == 1 || sample.rows == 1);
        if(sample.cols == 1) {
            cv::transpose(sample,tmp);
            tmp.copyTo( trainData.row( i ));
        }
        else if(sample.rows == 1) {
            sample.copyTo(trainData.row( i ));
        }
        i++;
    }
	
}

void TrainHog::saveLabeledMat(cv::Mat data)
{
	cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
	fs << "samples"<< data;
	fs.release();
	std::ofstream output_file("./labels.txt");
	std::ostream_iterator<int> output_iterator(output_file, "\n");
	std::copy(labels.begin(), labels.end(), output_iterator);
}

TrainHog::TrainHog()
{
	this->maxIterations = 3300;
	this->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	this->kernel = cv::ml::SVM::LINEAR;
	this->type = cv::ml::SVM::NU_SVC;
	this->epsilon = 1.e-6;
	this->coef0 = 0.0;
	this->degree = 3;
	this->gamma = 0.1;
	this->nu = 0.1;
	this->p = 0.1;
	this->c = 0.1;
	this->classifierName = "96_16_8_8_9_01.yml";
	pedestrianSize = cv::Size(48, 96);

}

TrainHog::TrainHog(int maxIterations, int termCriteria, int kernel, int type, double epsilon, double coef0,
	int degree, double gamma, double nu, double p, double c, std::string classifierName)
{
	this->maxIterations = maxIterations;
	this->termCriteria = termCriteria;
	this->kernel = kernel;
	this->type = type;
	this->epsilon = epsilon;
	this->coef0 = coef0;
	this->degree = degree;
	this->gamma = gamma;
	this->nu = nu;
	this->p = p;
	this->c = c;
	this->classifierName = classifierName;
	pedestrianSize = cv::Size(48, 96);
}

void TrainHog::printSettings()
{
	std::cout << std::setw(10) << std::setfill('_') << std::endl;
	std::cout << "SVM SETTING" << std::endl;
	std::cout << "MAX ITER: " << this->maxIterations << std::endl;
	std::cout << "TERM CRIT: " << this->termCriteria << std::endl;
	std::cout << "KERNEL: " << this->kernel << std::endl;
	std::cout << "TYPE SVM: " << this->type << std::endl;
	std::cout << "EPSILON: " << this->epsilon << std::endl;
	std::cout << "COEF0:" << this->coef0 << std::endl;
	std::cout << "DEGREE: " << this->degree << std::endl;
	std::cout << "GAMMA: " << this->gamma << std::endl;
	std::cout << "NU: " << this->nu << std::endl;
	std::cout << "P: " << this->p << std::endl;
	std::cout << "C: " << this->c << std::endl;
	std::cout << std::setw(10) << std::setfill('_') << std::endl;
}
