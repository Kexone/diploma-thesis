#include "trainhog.h"

void TrainHog::fillVectors(std::string &path, bool isNeg)
{
    assert(!path.empty());

    std::vector<cv::Mat> data;
    int posCount =0, negCount =0;
    int winSize = 96;
    cv::Mat frame;
	std::fstream sampleFile(path);
	std::string oSample;
	while (sampleFile >> oSample) {

		frame = cv::imread(oSample, CV_32FC3);
		cv::resize(frame, frame, cv::Size(winSize, winSize));
		data.emplace_back(frame.clone());
		if (!isNeg) {
			posCount++;
			labels.push_back(1);
		}
		else {
			negCount++;
			labels.push_back(0);
		}
	}
    if(!isNeg)
        posSamples = data;
    else
        negSamples = data;
    data.clear();



}

bool TrainHog::train()
{
    std::vector< cv::Mat > gradientLst;
    cv::Mat trainMat;
    extractFeatures(posSamples, gradientLst);
    extractFeatures(negSamples, gradientLst);
    convertSamples2Mat(gradientLst, trainMat);
    trainSvm(trainMat, labels);
    return true;
}

void TrainHog::extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst)
{
    cv::HOGDescriptor hog(
                    cv::Size(winSize,winSize), //winSize
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
        gradientLst.emplace_back( cv::Mat( descriptors ).clone() );
    }
}

void TrainHog::trainSvm(cv::Mat &trainMat, const std::vector<int> &labels)
{
    std::cout << "START training ...\n";
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setCoef0(0.0);
    svm->setDegree(3);
    int iterat = 3300;
    double epsilon = 1.e-6;
    svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, iterat, epsilon));
    svm->setGamma(0.1);
    svm->setKernel(cv::ml::SVM::INTER);
    svm->setNu(0.1);
    //svm->setP(0.1);
    svm->setC(0.1);
    svm->setType(cv::ml::SVM::NU_SVC);
    svm->train(trainMat, cv::ml::ROW_SAMPLE, cv::Mat(labels));
    svm->save(classifierName);

    std::cout << "training DONE ...\n";

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

TrainHog::TrainHog()
{

}
