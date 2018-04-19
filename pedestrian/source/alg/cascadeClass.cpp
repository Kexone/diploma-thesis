#include "cascadeClass.h"


CascadeClass::CascadeClass()
{
}

CascadeClass::CascadeClass(std::string filename)
{
	if (filename.compare("default")) {
		clasifier = cv::CascadeClassifier();
		clasifier.load(filename);
	}
	else
	{
		clasifier = cv::CascadeClassifier();
	//	clasifier.load();
		std::cout << clasifier.load("D://Codes//school//backupDT//repo//pedestrian//source//cascades//lbpcascades//case.xml");
		std::cout << "Initialized LBP cascades case" << std::endl;
	}	

}


void CascadeClass::detect(std::vector<CroppedImage>& frames, std::vector< std::vector < cv::Rect > > &rects)
{

	rects = std::vector<std::vector<cv::Rect>>(frames.size());
	if (frames.empty())
		return;
	for (uint x = 0; x < frames.size(); x++) {
		std::vector<cv::Rect> rRect;
		std::vector<cv::Rect> found;
		cv::Mat test = frames[x].croppedImg;

		assert(!test.empty());
		test.convertTo(test, CV_8UC1);
		cv::cvtColor(test, test, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(test, test);

		clasifier.detectMultiScale(test,	// image
			found,							// found <rect>
			1.1,							// scale factor
			2,								// min neighbors
			0 | cv::CASCADE_SCALE_IMAGE,	// flags
			cv::Size(30, 30)				// min size Size(30, 30)
		//	cv::Size(14,36)				// min size Size(30, 30)
		);
		if (found.empty()) {
			continue;
		}
		size_t i, j;
		for (i = 0; i< found.size(); i++)
		{
			cv::Rect r = found[i];
			rects[x].push_back(r);
			cv::rectangle(test, found[i].tl(), found[i].br(), cv::Scalar(0, 0, 255), 4, 8, 0);
		}
		cv::imshow("test", test);
	}
}

void CascadeClass::detect(cv::Mat& frame, std::vector<cv::Rect>& rects)
{
	// fflush(stdout);
	rects.clear();
	assert(!frame.empty());
	
	clasifier.detectMultiScale(frame,	// image
		rects,							// found <rect>
		1.1,							// scale factor
		2,								// min neighbors
		0 | cv::CASCADE_SCALE_IMAGE,	// flags
		cv::Size(30, 30)				// min size Size(30, 30)
	);

	cv::imshow("hog", frame);
}
