#include "pipeline.h"
#include <sstream>
#include <ctime>

int Pipeline::allDetections = 0;

Pipeline::Pipeline()
{
	//hog = Hog("48_96_16_8_8_9_01.yml");

}

void Pipeline::executeImages(std::string testSamplesPath)
{
	allDetections = 0;
	assert(!testSamplesPath.empty());
	cv::Mat frame;
	std::fstream sampleFile(testSamplesPath);
	std::string oSample;
	while (sampleFile >> oSample) {
		frame = cv::imread(oSample, CV_32FC3);
        if(frame.empty()) {
			sampleFile.close();
            break;
        }
       processStandaloneIm(frame);
	   cv::waitKey(0);
       frame.release();
    }
    cv::destroyWindow("Result");
}

void Pipeline::execute(int cameraFeed = 99)
{
	allDetections = 0;
    vs = new VideoStream(cameraFeed);
	std::cout << "Camera initialized." << std::endl;
	vs->openCamera();
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty()) {
            break;
        }
        process(frame);
        frame.release();
		cv::waitKey(5);
    }
  //  cv::destroyWindow("Test");
}

std::stringstream ss;

void Pipeline::execute(std::string cameraFeed)
{
	allDetections = 0;
    vs = new VideoStream(cameraFeed);
    vs->openCamera();
	std::cout << "Videostream initialized." << std::endl;
    int i = 0;
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty()) {
			delete vs;
			break;
        }
        process(frame);
        frame.release();
        
		cv::waitKey(5);
		ss << "../img/mat_" << i << ".jpg";
		cv::imwrite(ss.str(), localFrame);
		ss.str("");
		ss.clear();
		localFrame.release();
		i++;
    }
     cv::destroyWindow("Test");
}

void Pipeline::process(cv::Mat &frame)
{
	localFrame = frame.clone();
	preprocessing(frame);
	frame = mog.processMat(frame);
	preprocessing(frame, true);
	///cv::blur(frame, frame, cv::Size(9, 9));
	
	cv::imshow("MOG", frame);
	//cv::imwrite("test.jpg", frame);
	std::vector< cv::Rect > rect;// = ch.wrapObjects(localFrame, frame);

	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.emplace_back(CroppedImage(i, localFrame.clone(), rect[i]));
		}
		std::vector < std::vector < cv::Rect > > foundRect;
		
	//	foundRect = fhog.detect(croppedImages);
		foundRect = hog.detect(croppedImages);
		//foundRect = cc.detect(croppedImages);
		draw2mat(croppedImages, foundRect);
	}
	// if(Settings::showVideoFrames)
	//cv::imshow("Result", localFrame);
	frame.release();
	rect.clear();
}


void Pipeline::processStandaloneIm(cv::Mat &frame)
{
	localFrame = frame.clone();
	preprocessing(frame);
	std::vector < cv::Rect  > foundRect;
	//foundRect = fhog.detect(frame);
		foundRect = hog.detect(frame);
	//foundRect = cc.detect(frame);
	draw2mat(foundRect);

	// if(Settings::showVideoFrames)
	cv::imshow("Result", localFrame);
	foundRect.clear();
}

void Pipeline::preprocessing(cv::Mat& frame, bool afterMog)
{
	
	if(afterMog)
	{
		int dilation_type = cv::MORPH_RECT;
		int erosion_type = cv::MORPH_CROSS;
		int dilation_size = 4;
		int erosion_size = 3;
		cv::Mat dilMat = getStructuringElement(dilation_type,
			cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			cv::Point(dilation_size, dilation_size));
		cv::Mat eroMat = getStructuringElement(erosion_type,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		cv::dilate(frame, frame, dilMat);
		cv::erode(frame, frame, eroMat);
	}
	else
	{
		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		frame.convertTo(frame, CV_8UC1);

	}
	//cv::medianBlur(frame, frame, 3);
	
	//cv::Mat dst(frame.rows, frame.cols, CV_8UC1);
	//cv::bilateralFilter(frame, dst, 10, 1.5, 1.5, cv::BORDER_DEFAULT);
	//cv::GaussianBlur(frame, frame, cv::Size(9, 9), 2,4 , cv::BORDER_DEFAULT);
	//cv::blur(frame, frame, cv::Size(6, 6)); //medianBlur
	//cv::imshow("Blur", frame);
}

void Pipeline::draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect)
{
	for (uint j = 0; j < rect.size(); j++) {
		for (uint i = 0; i < rect[j].size(); i++) {
			cv::Rect r = rect[j][i];
			r.x += cvRound(croppedImages[j].offsetX);
			//r.width = cvRound(croppedImages[j].croppedImg.cols);
			r.y += cvRound(croppedImages[j].offsetY);
			//r.height = cvRound(croppedImages[j].croppedImg.rows);
			cv::rectangle(localFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
		}
		allDetections += rect[j].size();
	}
	rect.clear();

}

void Pipeline::draw2mat(std::vector < cv::Rect > &rect)
{
	for (uint i = 0; i < rect.size(); i++) {
		cv::rectangle(localFrame, rect[i], cv::Scalar(0, 255, 0), 3);
	}
	allDetections += rect.size();
	rect.clear();
}
