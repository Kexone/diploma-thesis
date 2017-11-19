#include "trainhog.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
void TrainHog::fillVectors(std::string &path, bool isNeg)
{
    assert(!path.empty());
	pedestrianSize = cv::Size(48,96);
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

	if (saveData) saveMatWithLabs(trainMat);

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
    svm->setCoef0(0.0);
    svm->setDegree(3);
    int iterat = 3300;
    double epsilon = 1.e-6; //6
    svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, iterat, epsilon));
    svm->setGamma(0.1); //0,1
    svm->setKernel(cv::ml::SVM::INTER);//linear
    svm->setNu(0.1); //0.1
    svm->setP(0.1);
    svm->setC(0.1); //0,1
    svm->setType(cv::ml::SVM::NU_SVC);
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

void TrainHog::saveMatWithLabs(cv::Mat data)
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

}
