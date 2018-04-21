#include "hog.h"
#include "../settings.h"
#if BAD_SAMPLES
#include <random>
#endif

Hog::Hog()
{

}

Hog::Hog(std::string svmPath)
{
	if (svmPath.compare("default")) {
		_hog = cv::HOGDescriptor(
			Settings::pedSize, //winSize  //def
			cv::Size(Settings::blockSize, Settings::blockSize), //,blocksize //def
			cv::Size(Settings::strideSize, Settings::strideSize), //blockStride // def
			cv::Size(Settings::cellSize, Settings::cellSize), //cellSize, //def
			9,		//nbins,
			0,		//derivAper,
			-1,		//winSigma,
			0,		//histogramNormType,
			0.2,	//L2HysThresh,
			1		//gamma corRection,
					//nlevels=64
		);
		_hog.svmDetector.clear();
		std::vector< float > hogDetector;
		_svm = cv::Algorithm::load<cv::ml::SVM>(svmPath);
		getSvmDetector(_svm, hogDetector);
		_hog.setSVMDetector(hogDetector);
		std::cout << "Initialized custom SVM " << svmPath << " size " << hogDetector.size() << std::endl;
		hogDetector.clear();
	}
	else
	{
		_hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		std::cout << "Initialized default people detector" << std::endl;
	}
	_hog.gammaCorrection = true;
}

//detection on cropped frames
void Hog::detect(std::vector<CroppedImage>& frames, std::vector< std::vector < cv::Rect > > &rects, std::vector < std::vector < float > > &distances) {

	//distances.clear();
	rects.clear();
	rects = std::vector < std::vector < cv::Rect > >(frames.size());
	distances = std::vector < std::vector < float > > (frames.size());
	for (size_t x = 0; x < frames.size(); x++) {
		rects[x].clear();
		std::vector<cv::Rect> found;
		cv::Mat test = frames[x].croppedImg;
		assert(!test.empty());
#if MY_DEBUG
			cv::imshow("test", test);
#endif
			if (Settings::cropHogBlurFilter.width != 0)
			cv::blur(test, test, Settings::cropHogBlurFilter);
		_hog.detectMultiScale(
			test,								// img
			found,								// foundLocation
			Settings::cropHogHitTreshold,       // hitThreshold
			Settings::cropHogWinStride,			// winStride
			Settings::cropHogPadding,			// padding
			Settings::cropHogScale,				// scale
			Settings::cropHogFinalTreshold,		// finalThreshold
			Settings::cropHogMeanshiftGrouping  // use meanshift grouping
		);

		if (found.empty()) {
			continue;
		}
		cv::groupRectangles(found, Settings::cropHogGroupTreshold, Settings::cropHogEps);
		//std::cout << (predicted) << std::endl;
		//float confidence = 1.0 / (1.0 + exp(-predict(test(found[0]), cv::ml::StatModel::Flags::RAW_OUTPUT)));
		//std::cout << confidence << std::endl;
	//	std::cout << 1.0f / (1.0f + std::exp(predict(test(found[0]), cv::ml::StatModel::Flags::RAW_OUTPUT))) << std::endl;
		if (found.size() > 1) {
			for (size_t i = 0; i < found.size(); i++) {
				for (size_t j = 0; j < found.size(); j++) {
					//if (i == j) continue;
					if ((found[i] & found[j]).area() >= found[i].area()/2) {
						found.erase(found.begin() + j);
						break;
					}
				}
			}
		}
		for (size_t i = 0; i< found.size(); i++)	{
#if BAD_SAMPLES
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, 1000);
			char imgName[30];
			std::sprintf(imgName, "bad/%d_negSample_%d.jpg", frames[x].id, dis(gen));
			cv::Mat cropped(test(found[i]));
			cv::imwrite(imgName, cropped);
#endif
			if (found[i].area() > Settings::cropHogMinArea) {
				rects[x].push_back(found[i]);
				cv::Mat cropped(test(found[i]));
				distances[x].push_back(getDistance(cropped));
			}
		}
		test.release();
		found.clear();
	}
}

//detection on full frame
void Hog::detect(cv::Mat& frame, std::vector < cv::Rect > &rects) {
	assert(!frame.empty());
	rects.clear();
	if(Settings::hogBlurFilter.width !=0)
		cv::blur(frame, frame, Settings::hogBlurFilter);
	//cv::medianBlur(frame, frame, 3);
	_hog.detectMultiScale(
		frame,							// img
		rects,							// foundLocation
		Settings::hogHitTreshold,		// hitThreshold
		Settings::hogWinStride,			// winStride
		Settings::hogPadding,			// padding
		Settings::hogScale,				// scale
		Settings::hogFinalTreshold,		// finalThreshold
		Settings::hogMeanshiftGrouping	// use meanshift grouping
	);
	if (rects.empty()) {
		return;
	}
	std::vector< int > weights;
	cv::groupRectangles(rects, weights, Settings::hogGroupTreshold, Settings::hogEps);
	std::vector<cv::Rect> found;

	for (size_t i = 0; i<rects.size(); i++)
	{
		if (rects[i].area() > Settings::hogMinArea)
			found.push_back(rects[i]);
#if BAD_SAMPLES
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, 1000);
		char imgName[30];
		std::sprintf(imgName, "bad/%d_posSample_%d.jpg", i, dis(gen));
		cv::Mat cropped(frame(rects[i]));
		cv::imwrite(imgName, cropped);
#endif
	}
	rects = found;
}

void Hog::detect(std::vector<cv::Mat> testLst, int &nTrue, int &nFalse, bool pedestrian)
{
	nTrue = 0;
	nFalse = 0;
	_hog.winSize = Settings::pedSize;
	for(auto &mat : testLst)	{
		int predicted = predict(mat);
		if(pedestrian)	{
			if (predicted == 1) nTrue++;
			else nFalse++;
		}	else  {
			if (predicted == 0) nTrue++;
			else nFalse++;
		}
	}
}

void Hog::getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector )
{
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

 //  CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
//   CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
   //            (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
}

float Hog::getDistance(cv::Mat img)
{
	float dist = predict(img, true); //cv::ml::StatModel::RAW_OUTPUT
	return 1.0f / (1.0f + std::exp(-dist));
}

float Hog::predict(cv::Mat img, int flags)
{
	std::vector< float > descriptors;
	cv::resize(img, img, Settings::pedSize);
	_hog.compute(img, descriptors, Settings::pedSize, cv::Size(0, 0));
	return _svm->predict(descriptors, cv::noArray(), flags);
}


