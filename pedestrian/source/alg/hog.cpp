#include "hog.h"
#include <random>
#include "../settings.h"


Hog::Hog()
{

}

Hog::Hog(std::string svmPath)
{
	if (svmPath.compare("default")) {
		hog = cv::HOGDescriptor(
			Settings::pedSize, //winSize  //def
			cv::Size(Settings::blockSize, Settings::blockSize), //,blocksize //def
			cv::Size(Settings::strideSize, Settings::strideSize), //blockStride // def
			cv::Size(Settings::cellSize, Settings::cellSize), //cellSize, //def
			9, //nbins,
			0, //derivAper,
			-1, //winSigma,
			0, //histogramNormType,
			0.2, //L2HysThresh,
			0 //gammal corRection,
			  //nlevels=64
		);
		hog.svmDetector.clear();
		std::vector< float > hogDetector;
		svm = cv::Algorithm::load<cv::ml::SVM>(svmPath);
		getSvmDetector(svm, hogDetector);
		hog.setSVMDetector(hogDetector);
		std::cout << "Initialized custom SVM " << svmPath << " size " << hogDetector.size() << std::endl;
	//	hogDetector.clear();
	}
	else
	{
		hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
		std::cout << "Initialized default people detector" << std::endl;
	}
	hog.gammaCorrection = true;
}


void Hog::detect(std::vector<CroppedImage>& frames, std::vector< std::vector < cv::Rect > > &rects, std::vector < std::vector < float > > &distances) {

	//distances.clear();
	rects.clear();
	rects = std::vector < std::vector < cv::Rect > > (frames.size());
	// fflush(stdout);
	for (size_t x = 0; x < frames.size(); x++) {
		//std::vector<cv::Rect> rRect;
		std::vector<cv::Rect> found;
		cv::Mat test = frames[x].croppedImg;
		assert(!test.empty());
		test.convertTo(test, CV_8UC3);

	//	cv::cvtColor(test, test, CV_BGR2GRAY);
	//	cv::equalizeHist(test, test);
		//cv::resize(test, test, cv::Size(test.cols*1.5, test.rows*1.5));
#if MY_DEBUG
		cv::imshow("test", test);
#endif
		//cv::waitKey(15);

		hog.detectMultiScale(
			test,					// testing img
			found,					// foundLocation <rect>
			0,						// hitThreshold = 0 // 1
			cv::Size(8, 8),			// winStride size(8, 8)
			cv::Size(0, 0),			// padding size(0, 0)
			1.05,					// scale = 1,05
			1,	/* 1*/					// finalThreshold = 2 // 0
			false					// use meanshift grouping = false
		);
		
	//		std::cout << found.size() << std::endl;

		if (found.empty()) {
			continue;
		}

		//std::cout << (predicted) << std::endl;
		//float confidence = 1.0 / (1.0 + exp(-predict(test(found[0]), cv::ml::StatModel::Flags::RAW_OUTPUT)));
		//std::cout << confidence << std::endl;
	//	std::cout << 1.0f / (1.0f + std::exp(predict(test(found[0]), cv::ml::StatModel::Flags::RAW_OUTPUT))) << std::endl;
		size_t i, j;
	
		for (i = 0; i< found.size(); i++)
		{
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<> dis(0, 1000);
			char imgName[30];
			std::sprintf(imgName, "bad/%d_negSample_%d.jpg", frames[x].id, dis(gen));
			cv::Mat cropped(test(found[i]));
			cv::imwrite(imgName, cropped);
			cv::Rect r = found[i];

			//for (j = 0; j<found.size(); j++)
			//    if (j != i && (r & found[j]) == r)
			//          break;
			//    if (j == found.size())
		//	found[i].x /= 1.5;
			//found[i].y /= 1.5;
			rects[x].push_back( found[i] );
			//distances[x].push_back( getDistance(cropped) );
			//  std::cout << "TL" << found[i].tl().x << found[i].tl().y << " BR" << found[i].br().x << found[i].br().y;
			//          cv::rectangle(test, found[i].tl(), found[i].br(),cv::Scalar(0,0,255),4,8,0);
		}
		test.release();
		found.clear();
	}
}

void Hog::detect(cv::Mat& frame, std::vector < cv::Rect > &rects) {
	std::vector<cv::Rect> found_filtered;
	fflush(stdout);
	rects.clear();
	assert(!frame.empty());
	cv::Mat gray;
	frame.copyTo(gray);
	gray.convertTo(gray, CV_8UC1);
	cv::cvtColor(gray, gray, CV_BGR2GRAY);
	//cv::threshold(frame, frame, 80, 255, cv::THRESH_BINARY);
//	cv::equalizeHist(gray, gray);
//	hogDetectMultiScale(frame, rects);
	hog.detectMultiScale(
		frame,					// testing img
		rects,					// foundLocation <rect>
		2.2,						// hitThreshold = 0 // 1
		cv::Size(8, 8),			// winStride size(8, 8)
		cv::Size(0, 0),			// padding size(0, 0)
		1.07,					// scale = 1,05
		2.0,//4.0,	/* 1*/				// finalThreshold = 2 // 0
		false					// use meanshift grouping = false
	);
	std::cout << rects.size();
//	if (rects.size() > 0) std::cout << " " <<  getDistance(frame(rects[0])) << std::endl;

//	cv::groupRectangles(rects, 0.79,5.72);
	std::cout << "  " <<  rects.size() << std::endl;
	size_t i, j;
	for (i = 0; i<rects.size(); i++)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> dis(0, 1000);
		char imgName[30];
		std::sprintf(imgName, "bad/%d_negSample_%d.jpg", i, dis(gen));
		cv::Mat cropped(frame(rects[i]));

		cv::imwrite(imgName, cropped);


		}
////		cv::Rect r = rects[i];
////		for (j = 0; j<rects.size(); j++)
////			if (j != i && (r & rects[j]) == r)
////				break;
////		if (j == rects.size())
////			found_filtered.push_back(r);
////	}
////
////	for (i = 0; i<found_filtered.size(); i++)
////	{
////		cv::Rect r = found_filtered[i];
////		r.x += cvRound(r.width*0.1);
////		r.width = cvRound(r.width*0.8);
////		r.y += cvRound(r.height*0.07);
////		r.height = cvRound(r.height*0.8);
////		rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
////	}
////	
////	cv::imshow("hog", frame);
////	rects = found_filtered;
	//cv::cvtColor(test, test, CV_BGR2GRAY);
		//cv::equalizeHist(test, test);
//	cv::imshow("hog", frame);
}

void Hog::detect(std::vector<cv::Mat> testLst, int &nTrue, int &nFalse, bool pedestrian)
{
	nTrue = 0;
	nFalse = 0;
	hog.winSize = Settings::pedSize;
	for(auto &mat : testLst)
	{
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

void Hog::hogDetectMultiScale(cv::Mat img, std::vector<cv::Rect>& found)
{
	hog.detectMultiScale(
		img,					// testing img
		found,					// foundLocation <rect>
		0.1,						// hitThreshold = 0 // 1
		cv::Size(8, 8),			// winStride size(8, 8)
		cv::Size(0, 0),			// padding size(0, 0)
		1.01,					// scale = 1,05
		1,	/* 1*/					// finalThreshold = 2 // 0
		false					// use meanshift grouping = false
	);
}

float Hog::getDistance(cv::Mat img)
{
	float dist = predict(img, true); //cv::ml::StatModel::RAW_OUTPUT
	return 1.0f / (1.0f + std::exp(dist));
}

float Hog::predict(cv::Mat img, int flags)
{
	std::vector< float > descriptors;
	hog.compute(img, descriptors, Settings::pedSize, cv::Size(0, 0));
	return svm->predict(descriptors, cv::noArray(), flags);
}


