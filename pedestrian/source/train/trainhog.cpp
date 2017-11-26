#include "trainhog.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
void TrainHog::fillVectors(std::string path, bool isNeg)
{
    assert(!path.empty());
    std::vector<cv::Mat> data;
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
	}
    if(!isNeg)
        posSamples = data;
    else
        negSamples = data;
    data.clear();


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

	std::cout << "Positive samples: " << posSamples.size() << std::endl;
	std::cout << "Negative samples: " << negSamples.size() << std::endl;

    extractFeatures(posSamples, gradientLst);
	posSamples.clear();
    extractFeatures(negSamples, gradientLst);
	negSamples.clear();
    convertSamples2Mat(gradientLst, trainMat);
	gradientLst.clear();

	if (saveData) saveLabeledMat(trainMat);

    trainSvm(trainMat, labels);
}

void GammaCorrection(cv::Mat src, cv::Mat& dst, float fGamma)
{
	unsigned char lut[256];

	for (int i = 0; i < 256; i++)
	{
		lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
		{
			cv::MatIterator_<uchar> it, end;

			for (it = dst.begin<uchar>() , end = dst.end<uchar>(); it != end; it++)
				*it = lut[(*it)];
			break;
		}
	case 3:
		{
			cv::MatIterator_<cv::Vec3b> it, end;
			for (it = dst.begin<cv::Vec3b>() , end = dst.end<cv::Vec3b>(); it != end; it++)

			{
				(*it)[0] = lut[((*it)[0])];
				(*it)[1] = lut[((*it)[1])];
				(*it)[2] = lut[((*it)[2])];
			}
			break;
		}
	}
}

cv::Mat featureSobel(const cv::Mat& mat, int minThreshold) {
	cv::Mat src = mat.clone();
	cv::Mat srcGray, gradX, gradY, grad;

	// Checks
	assert(src.type() == CV_8UC3); // 8UC3

								   // Convert to gray and blur
	cv::cvtColor(src, srcGray, CV_BGR2GRAY);
	cv::medianBlur(srcGray, srcGray, 3);

	/// Gradient X & Y
	Sobel(srcGray, gradX, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	Sobel(srcGray, gradY, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

	// Convert to 8UC1
	convertScaleAbs(gradX, gradX);
	convertScaleAbs(gradY, gradY);

	// Checks
	assert(gradX.type() == CV_8UC1); // 8UC1
	assert(gradY.type() == CV_8UC1); // 8UC1
	assert(grad.type() == CV_8UC1); // 8UC1

									/// Total Gradient (approximate)
	addWeighted(gradX, 0.5, gradY, 0.5, 0, grad);


	return grad;
}

void sum(std::vector<float> &vec, float &channelsValues)
{
	float sum = 0;
	for (auto& n : vec)
		sum += n;
	channelsValues = sum;
}
cv::Mat featureColorGradient(const cv::Mat &mat, cv::HOGDescriptor hog) {
	float channelsValues[3];
	std::vector < cv::Mat > channels(3);
	std::vector < std::vector < float > > descriptors(3);

	cv::split(mat, channels);
	for (int i = 0; i < channels.size(); i++)
	{
		hog.compute(channels[i], descriptors[i], cv::Size(8, 8), cv::Size(0, 0));
		sum(descriptors[i], channelsValues[i]);
	}
	return cv::Mat(descriptors[std::distance(channelsValues, std::find(channelsValues, channelsValues + 5, *std::max_element(channelsValues, channelsValues + 3)))]);
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
		GammaCorrection(mat, gr,0.5);
		//mat.convertTo(gr, CV_64F);
	//	cv::imshow("bef", gr);
        //cv::cvtColor(mat, gr,cv::COLOR_BGR2GRAY);
		//cv::pow(gr,1.1, gr);
	//	cv::convertScaleAbs(gr, gr, 1, 0);
		//cv::imshow("bef", gr);
		//gr = featureColorGradient(gr);
		//cv::imshow("gra", gr);
		gr.convertTo(gr, CV_8U);
		//cv::imshow("aft", gr);
		//cv::waitKey(25);

		//hog.compute(gr, descriptors,cv::Size(8, 8),cv::Size(0, 0));
		//gradientLst.push_back(cv::Mat(descriptors).clone());
		//gradientLst.push_back(cv::Mat(featureSobel(mat, 80)));
		gradientLst.push_back( featureColorGradient(gr,hog).clone() );
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
	this->maxIterations = 400; // 3300;
	this->termCriteria = CV_TERMCRIT_ITER + CV_TERMCRIT_EPS;
	this->kernel = cv::ml::SVM::LINEAR;
	this->type = cv::ml::SVM::C_SVC;
	this->epsilon = 1.e-6;
	this->coef0 = 0.0;
	this->degree = 3;
	this->gamma = 0.1;
	this->nu = 0.1; //0.1;
	this->p = 0.0369121;
	this->c = 0.01;
	this->classifierName = "48_96_16_8_8_9_01.yml";
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
