#include "pipeline.h"
#include <sstream>
#include <ctime> // @DEBUG

#define PURE_HOG 1
#define MIXTURED_HOG 2
#define PURE_FHOG 3
#define MIXTURED_FHOG 4
#define PURE_CASCADE 5
#define MIXTURED_CASCADE 6

int Pipeline::allDetections = 0;


Pipeline::Pipeline(std::string svmPath, int algType)
{
	_typeAlgorithm = algType;
	if (_typeAlgorithm == PURE_HOG || _typeAlgorithm == MIXTURED_HOG)
		_hog = Hog(svmPath);
	else if (_typeAlgorithm == PURE_FHOG || _typeAlgorithm == MIXTURED_FHOG)
		_fhog = new FHog("data.dat");
	else if (_typeAlgorithm == PURE_CASCADE || _typeAlgorithm == MIXTURED_CASCADE)
		_cc = CascadeClass(svmPath);
	if (_typeAlgorithm == MIXTURED_HOG || _typeAlgorithm == MIXTURED_FHOG || _typeAlgorithm == MIXTURED_CASCADE) {
		_dilMat = getStructuringElement(_dilation_type,
			cv::Size(2 * _dilation_size + 1, 2 * _dilation_size + 1),
			cv::Point(_dilation_size, _dilation_size));
		_eroMat = getStructuringElement(_erosion_type,
			cv::Size(2 * _erosion_size + 1, 2 * _erosion_size + 1),
			cv::Point(_erosion_size, _erosion_size));
	}
	allDetections = 0;

}

#if MY_DEBUG
std::vector< std::vector<cv::Rect> > trained;//@DEBUG
std::vector< std::vector<cv::Rect> > tested;//@DEBUG
int test;//@DEBUG
#endif

//	Execute for images
void Pipeline::executeImages(std::string testSamplesPath)
{
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
       processStandaloneImage(frame);
	   cv::waitKey(0);
       frame.release();
    }
    cv::destroyWindow("Result");
}

//	Execute for webcam stream
void Pipeline::execute(int cameraFeed = 99)
{
	_vs = new VideoStream(cameraFeed);
	std::cout << "Camera initialized." << std::endl;
	_vs->openCamera();
    for(int i = 0; ; i++ ) {
        cv::Mat frame = _vs->getFrame();
        if(frame.empty()) {
            break;
        }
		//  process(frame, i);
        frame.release();
		cv::waitKey(5);
    }
  //  cv::destroyWindow("Test");
}

// Execute for video stream
void Pipeline::execute(std::string cameraFeed)
{
	_vs = new VideoStream(cameraFeed);
	_vs->openCamera();
	std::cout << "Videostream initialized." << std::endl;
	_rects2Eval = std::vector < std::vector < std::vector < cv::Rect > > >(_vs->totalFrames);
	_distances = std::vector < std::vector < std::vector < float > > >(_vs->totalFrames);

#if MY_DEBUG
	loadRects(Settings::nameTrainedFile, trained); // @DEBUG
	loadRects(Settings::nameFile, tested); // @DEBUG
#endif

	for (int i = 0; ; i++)	{
#if MY_DEBUG
		test = i; //@DEBUG
#endif
		cv::Mat frame = _vs->getFrame();
		if (frame.empty())	{
			delete _vs;
			saveResults();
			break;
		}
		//time_t time = clock(); // @DEBUG
		if (_typeAlgorithm == PURE_HOG)
			pureHoG(frame, i);
		else if (_typeAlgorithm == MIXTURED_HOG)
			mogAndHog(frame, i);
		else if (_typeAlgorithm == PURE_FHOG)
			pureFHoG(frame, i);
		else if (_typeAlgorithm == MIXTURED_FHOG)
			mogAndFHog(frame, i);
		else if (_typeAlgorithm == PURE_CASCADE)
			pureCascade(frame, i);
		else if (_typeAlgorithm == MIXTURED_CASCADE)
			mogAndCascade(frame, i);

		//time = clock() - time; // @DEBUG 
		//	std::cout << static_cast<float>(time) / CLOCKS_PER_SEC << std::endl; // @DEBUG
		cv::waitKey(5);
		if (Settings::showVideoFrames)
			cv::imshow("Result", _localFrame);
		cv::waitKey(5);
		frame.release();

#if MY_DEBUG
		std::stringstream ss;
		ss << "img/mat_" << i << ".jpg";
		cv::imwrite(ss.str(), _localFrame);
		ss.str("");
		ss.clear();
#endif
		_localFrame.release();
	}
}

void Pipeline::mogAndHog(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);
	_mog.processMat(frame);
	dilateErode(frame);

	std::vector< cv::Rect > rect;
	_ch.wrapObjects(frame, rect);

	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.emplace_back(CroppedImage(i, frame.clone(), rect[i]));
		}
		std::vector < std::vector < cv::Rect > > foundRect;
		std::vector < std::vector < float > > distances(croppedImages.size());

		_hog.detect(croppedImages, foundRect, distances);
		_distances[cFrame] = distances;
		rectOffset(foundRect, croppedImages, _rects2Eval[cFrame]);
		draw2mat(croppedImages, foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::pureHoG(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_hog.detect(frame, foundRect);
	draw2mat(foundRect);
	_rects2Eval[cFrame].push_back(foundRect);
	foundRect.clear();
	frame.release();
}

void Pipeline::mogAndFHog(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);
	_mog.processMat(frame);
	dilateErode(frame);

	std::vector< cv::Rect > rect;
	_ch.wrapObjects(frame, rect);

	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.emplace_back(CroppedImage(i, _localFrame.clone(), rect[i]));
		}
		std::vector < std::vector < cv::Rect > > foundRect;

		_fhog->detect(croppedImages, foundRect);
		rectOffset(foundRect, croppedImages, _rects2Eval[cFrame]);
		draw2mat(croppedImages, foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::pureFHoG(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_fhog->detect(frame, foundRect);
	draw2mat(foundRect);
	_rects2Eval[cFrame].push_back(foundRect);
	foundRect.clear();
	frame.release();
}

void Pipeline::processStandaloneImage(cv::Mat &frame)
{
	_localFrame = frame.clone();
	preprocessing(frame);
	std::vector < cv::Rect  > foundRect;
	//foundRect = fhog.detect(frame);
	_hog.detect(frame, foundRect);
	//foundRect = cc.detect(frame);
	draw2mat(foundRect);

	// if(Settings::showVideoFrames)
	cv::imshow("Result", _localFrame);
	foundRect.clear();
}

void Pipeline::pureCascade(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_cc.detect(frame, foundRect);
	draw2mat(foundRect);
	_rects2Eval[cFrame].push_back(foundRect);
	foundRect.clear();
	frame.release();
}

void Pipeline::mogAndCascade(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);
	_mog.processMat(frame);
	dilateErode(frame);

	std::vector< cv::Rect > rect;
	_ch.wrapObjects(frame, rect);

	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.emplace_back(CroppedImage(i, _localFrame.clone(), rect[i]));
		}
		std::vector < std::vector < cv::Rect > > foundRect;

		_cc.detect(croppedImages, foundRect);
		rectOffset(foundRect, croppedImages, _rects2Eval[cFrame]);
		draw2mat(croppedImages, foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::preprocessing(cv::Mat& frame)
{
	cv::cvtColor(frame, frame, CV_BGR2GRAY);
	frame.convertTo(frame, CV_8UC1);
	//cv::medianBlur(frame, frame, 3);

	//cv::Mat dst(frame.rows, frame.cols, CV_8UC1);
	//cv::bilateralFilter(frame, dst, 10, 1.5, 1.5, cv::BORDER_DEFAULT);
	//cv::GaussianBlur(frame, frame, cv::Size(9, 9), 2,4 , cv::BORDER_DEFAULT);
	//cv::blur(frame, frame, cv::Size(6, 6)); //medianBlur
	//cv::imshow("Blur", frame);
}

void Pipeline::dilateErode(cv::Mat& frame)
{
	cv::dilate(frame, frame, _dilMat);
	cv::erode(frame, frame, _eroMat);
}

void Pipeline::draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect)
{
	for (uint j = 0; j < rect.size(); j++) {
		for (uint i = 0; i < rect[j].size(); i++) {
			cv::Rect r = rect[j][i];

#if MY_DEBUG
			if (!trained[test].empty()) {//@DEBUG
				if (!tested[test].empty()) //@DEBUG
					cv::rectangle(localFrame, tested[test][0], cv::Scalar(0, 0, 255), 3);
				cv::rectangle(_localFrame, trained[test][0], cv::Scalar(0, 255, 255), 3);
				cv::rectangle(_localFrame, r & trained[test][0], cv::Scalar(255, 0, 0), 3);
				std::cout << (r & trained[test][0]).area() << std::endl;
			}
#endif

			cv::rectangle(_localFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
			
		}
		allDetections += rect[j].size();
	}
	rect.clear();

}

void Pipeline::draw2mat(std::vector < cv::Rect > &rect)
{
	for (uint i = 0; i < rect.size(); i++) {
		cv::rectangle(_localFrame, rect[i], cv::Scalar(0, 255, 0), 3);
	}
	allDetections += rect.size();
}

void Pipeline::saveResults()
{
	std::ofstream fs;
	fs.open(Settings::nameFile);
	fs << _rects2Eval.size() << std::endl;
	for (uint i = 0; i < _rects2Eval.size(); i++)	{
		for (uint j = 0; j < _rects2Eval[i].size(); j++)	{
			for (uint k = 0; k < _rects2Eval[i][j].size(); k++)	{
				if (_rects2Eval[i][j][k].area() == 0) continue;
				fs << i << " " << _rects2Eval[i][j][k].tl().x << " " << _rects2Eval[i][j][k].tl().y << " " << _rects2Eval[i][j][k].br().x << " " << _rects2Eval[i][j][k].br().y << std::endl;
			}
		}
	}
	fs.close();

	fs.open(Settings::nameFile + "_distances.txt");
	fs << "name of classificator\n";
	fs << _distances.size() << "\n";
	for (uint i = 0; i < _distances.size(); i++) {
		for (uint j = 0; j < _distances[i].size(); j++) {
			for (uint k = 0; k < _distances[i][j].size(); k++) {
				fs << i << " " << _distances[i][j][k] << std::endl;
			}
		}
	}
	fs.close();
}

void Pipeline::loadRects(std::string filePath, std::vector< std::vector<cv::Rect> > & rects)
{
	std::ifstream ifs;
	ifs.open(filePath);
	int x1, x2, y1, y2, cFrame;
	std::string frame;
	std::string fileContents;
	unsigned int curLine;
	int i = 0;
	if (ifs.is_open())
	{
		getline(ifs, fileContents, '\x1A');
		ifs.close();
	}
	std::getline(ifs, fileContents, '\n');
	std::istringstream iss(fileContents);
	iss >> curLine;
	rects = std::vector< std::vector<cv::Rect> >(curLine);
	while (!iss.eof()) {
		
		iss >> cFrame >> x1 >> y1 >> x2 >> y2;
		cv::Point p1(x1, y1);
		cv::Point p2(x2, y2);
		if(cFrame >= 0)
			rects[cFrame].emplace_back(cv::Rect(p1, p2));
	}
}

void Pipeline::rectOffset(std::vector<std::vector<cv::Rect>> &rects, std::vector< CroppedImage > &croppedImages, std::vector<std::vector<cv::Rect>> &rects2Save)
{
	for (uint j = 0; j < rects.size(); j++) {
		for (uint i = 0; i < rects[j].size(); i++) {
			rects[j][i].x += cvRound(croppedImages[j].offsetX);
			rects[j][i].y += cvRound(croppedImages[j].offsetY);
		}
		rects2Save.push_back(rects[j]);
	}
}

void Pipeline::evaluate()
{
	std::vector< std::vector<cv::Rect> > trained;
	std::vector< std::vector<cv::Rect> > test;
	loadRects(Settings::nameFile, test);
	loadRects(Settings::nameTrainedFile, trained);
	int truePos = 0, falsePos = 0;
	int trueNeg = 0, falseNeg = 0;
	for (int i = 0; i < trained.size(); i++) {
		//std::cout << i << " TESTING!" << std::endl;
	//	if (test[i].empty() && trained[i].empty())	{ trueNeg++;  continue; } // There is no pedestrian - OK
		if (test[i].empty() && !trained[i].empty()) { falsePos += trained[i].size(); continue; } // There is no pedestrian but something detect
		if (!test[i].empty() && trained[i].empty()) { falseNeg += test[i].size(); continue; } // There is pedestrian but no detected
		if (test[i].size() > trained[i].size())			{ falseNeg += test[i].size() - trained[i].size(); }
		for (int j = 0; j < test[i].size(); j++) {
			for(int k = 0; k < trained[i].size(); k++)	{
				float inter = static_cast<float>((trained[i][k] & test[i][j]).area());
				float uni = static_cast<float>((trained[i][k] | test[i][j]).area());
				//if ((trained[i][k] & test[i][j]).area()  > (trained[i][k].area() / 2)) // If intersect between detection and ground truth is at least 50 % of ground truth F1-0.855721
				if(static_cast<float>(inter / uni) >= 0.25f) //IoU  - Intersection over Union 
				{
					truePos++; // pedestrian founded
				}
				else
				{
					falsePos++; // detect something else than pedestrian
					falseNeg++; // pedestrian not founded
				}
			}
		}
	}
//	float acc = static_cast<float>(truePos + trueNeg) /
	//	static_cast<float>(truePos + trueNeg + falsePos + falseNeg);

	float f1score = static_cast<float>(2 * truePos) /
		static_cast<float>(2 * truePos + falsePos + falseNeg);

	float precision = static_cast<float>(truePos) / (truePos + falsePos); // Precision is the percentage true positives in the retrieved results.
	float recall = static_cast<float>(truePos) / (truePos + falseNeg); // Recall is the percentage of the pedestrians that the system retrieves.
	// ??% of the retrieved results were airplanes, and ??% of the airplanes were retrieved.

	std::cout << "True Positive: " << truePos << std::endl;
	std::cout << "False Positive: " << falsePos << std::endl;
	std::cout << "True Negative: " << trueNeg << std::endl;
	std::cout << "False Negative: " << falseNeg << std::endl;
//	std::cout << "Accuracy: " << acc << std::endl;
//	std::cout << "Accuracy (%): " << static_cast<int>(acc * 100) << std::endl;
	std::cout << "F1 score : " << f1score << std::endl;
	std::cout << "Precision : " << precision << std::endl;
	std::cout << "Recall: " << recall << std::endl;
}


