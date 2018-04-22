#include "pipeline.h"
#include <sstream>

#define PURE_HOG 1
#define MIXTURED_HOG 2
#define PURE_FHOG 3
#define MIXTURED_FHOG 4
#define PURE_CASCADE 5
#define MIXTURED_CASCADE 6


Pipeline::Pipeline(std::string svmPath, int algType): _vs(nullptr)
{
	int dilation_type = cv::MORPH_CROSS;
	int erosion_type = cv::MORPH_CROSS;
	int dilation_size = Settings::dilationSize;
	int erosion_size = Settings::erosionSize;
	_typeAlgorithm = algType;
	if (_typeAlgorithm == PURE_HOG || _typeAlgorithm == MIXTURED_HOG)
		_hog = Hog(svmPath);
	else if (_typeAlgorithm == PURE_FHOG || _typeAlgorithm == MIXTURED_FHOG)
		_fhog = new Fhog(svmPath);
	else if (_typeAlgorithm == PURE_CASCADE || _typeAlgorithm == MIXTURED_CASCADE)
		_cc = CascadeClass(svmPath);
	if (_typeAlgorithm == MIXTURED_HOG || _typeAlgorithm == MIXTURED_FHOG || _typeAlgorithm == MIXTURED_CASCADE)
	{
		_dilMat = getStructuringElement(dilation_type,
		                                cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		                                cv::Point(dilation_size, dilation_size));
		_eroMat = getStructuringElement(erosion_type,
		                                cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		                                cv::Point(erosion_size, erosion_size));
	}
}

#if MY_DEBUG
std::vector< std::vector<cv::Rect> > trained;
std::vector< std::vector<cv::Rect> > tested;
int test;
#endif

//	Execute for images
void Pipeline::executeImages(std::string testSamplesPath)
{
	assert(!testSamplesPath.empty());
	cv::Mat frame;
	std::fstream sampleFile(testSamplesPath);
	std::string oSample;
	//_rects2Eval = std::vector < std::vector < std::vector < cv::Rect > > >;
	int i = 0;
	while (sampleFile >> oSample) {
		std::cout << oSample << std::endl;
		frame = cv::imread(oSample, CV_LOAD_IMAGE_UNCHANGED);
        if(frame.empty()) {
			sampleFile.close();
            break;
        }
		_rects2Eval.resize(i + 1);
#if CALC_DIST
		_distances.resize(i+1);
#endif
       processStandaloneImage(frame,i++);
	   if (Settings::showVideoFrames)
		   cv::imshow("Result", _localFrame);
	   cv::waitKey(0);
       frame.release();
    }
	saveResults();
    cv::destroyWindow("Result");
}

//	Execute for webcam stream
void Pipeline::execute(int cameraFeed = 99)
{
	_vs = new VideoStream(cameraFeed);
	std::cout << "Camera initialized." << std::endl;
	_vs->openCamera();
	_rects2Eval = std::vector < std::vector < std::vector < cv::Rect > > >(1);
    for( ; ; ) {
        cv::Mat frame = _vs->getFrame();
        if(frame.empty()) {
            break;
        }
		if (_typeAlgorithm == PURE_HOG)
			pureHoG(frame, 0);
		else if (_typeAlgorithm == MIXTURED_HOG)
			mogAndHog(frame, 0);
		else if (_typeAlgorithm == PURE_FHOG)
			pureFHoG(frame, 0);
		else if (_typeAlgorithm == MIXTURED_FHOG)
			mogAndFHog(frame, 0);
		else if (_typeAlgorithm == PURE_CASCADE)
			pureCascade(frame, 0);
		else if (_typeAlgorithm == MIXTURED_CASCADE)
			mogAndCascade(frame, 0);

		cv::imshow("Result", _localFrame);
		cv::waitKey(10);
        frame.release();
		cv::waitKey(5);
    }
  //  cv::destroyWindow("Test");
}

// Execute for video stream
void Pipeline::execute(std::string cameraFeed)
{
//	cv::Mat test = cv::imread("D:/dlib/tools/imglab/build/Release/sudipDas/0.bmp");/ /@TODO
//	_fhog->predict(test, 0);
	_vs = new VideoStream(cameraFeed);
	_vs->openCamera();
	std::cout << "Videostream initialized." << std::endl;
	_rects2Eval = std::vector < std::vector < std::vector < cv::Rect > > >(_vs->totalFrames);
#if CALC_DIST
	_distances = std::vector < std::vector < std::vector < float > > >(_vs->totalFrames);
#endif

#if MY_DEBUG
	loadRects(Settings::nameTrainedFile, trained);
	loadRects(Settings::nameFile, tested);
#endif

	for (int i = 0; ; i++)	{
#if MY_DEBUG
		test = i;
#endif
		cv::Mat frame = _vs->getFrame();
		if (frame.empty())	{
			delete _vs;
			saveResults();
			break;
		}

#if SAVE_ROI_GT
		if (!trained[i].empty()) { //CROPPING PEDESTRIAN FROM GROUND TRUTH
			cv::Rect cropPed = cv::Rect(trained[i][0].tl().x - 0, trained[i][0].tl().y - 10, 64, 128);
			char imgName[30];
			std::sprintf(imgName, "bad/posSample_%d.jpg", i);
			cv::Mat cropped(frame(trained[i][0]));
			cv::imwrite(imgName, cropped);
		}
#endif

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

		if (Settings::showVideoFrames)
			cv::imshow("Result", _localFrame);
		cv::waitKey(2);
		frame.release();

#if SAVE_MAT
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
	std::vector< cv::Rect > rect;
	_localFrame = frame.clone();
	preprocessing(frame);
	_mog.processMat(frame);
	dilateErode(frame);
	_ch.wrapObjects(frame, rect);

#if MY_DEBUG
	cv::imshow("mog", frame);
#endif

	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		std::vector < std::vector < cv::Rect > > foundRect;
		std::vector < std::vector < float > > distances(croppedImages.size());

		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.push_back(CroppedImage(i, _localFrame.clone(), rect[i]));
		}
		_hog.detect(croppedImages, foundRect, distances);
#if CALC_DIST
		_distances[cFrame] = distances;
#endif
		rectOffset(foundRect, croppedImages, _rects2Eval[cFrame]);
		if (Settings::showVideoFrames)
			draw2mat(foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::pureHoG(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	//preprocessing(frame);

	std::vector < cv::Rect > foundRect;
	std::vector < float > distances;

	_hog.detect(frame, foundRect, distances);
	if (Settings::showVideoFrames)
		draw2mat(foundRect);
	if(!foundRect.empty() && cFrame >= 0)
		_rects2Eval[cFrame].push_back(foundRect);
	_distances[cFrame].push_back(distances);
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
		if (Settings::showVideoFrames)
			draw2mat(foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::pureFHoG(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	//preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_fhog->detect(frame, foundRect);
	if (Settings::showVideoFrames)
		draw2mat(foundRect);
	if (!foundRect.empty() && cFrame >= 0)
		_rects2Eval[cFrame].push_back(foundRect);
	foundRect.clear();
	frame.release();
}

void Pipeline::processStandaloneImage(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	std::vector < cv::Rect  > foundRect;
	std::vector < float > distances;
	_hog.detect(frame, foundRect, distances); 
	if (Settings::showVideoFrames)
		draw2mat(foundRect);

	_rects2Eval[cFrame].push_back(foundRect);
	_distances[cFrame].push_back(distances);
	foundRect.clear();
}

void Pipeline::pureCascade(cv::Mat &frame, int cFrame)
{
	_localFrame = frame.clone();
	preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_cc.detect(frame, foundRect);
	if (Settings::showVideoFrames)
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
		if (Settings::showVideoFrames)
			draw2mat(foundRect);
		foundRect.clear();
	}
	frame.release();
	rect.clear();
}

void Pipeline::preprocessing(cv::Mat& frame)
{
	frame.convertTo(frame, CV_8UC1);
	cv::cvtColor(frame, frame, CV_BGR2GRAY);
	cv::medianBlur(frame, frame,5); //GOOD

//	cv::Mat dst(frame.size(), CV_8UC1);
//	cv::bilateralFilter(frame, dst,10, 1, 1, cv::BORDER_DEFAULT); //TO SLOW
//	dst.copyTo(frame);

	//	cv::GaussianBlur(frame, frame, cv::Size(3,3),3.0,2.5 , cv::BORDER_DEFAULT);
	//cv::blur(frame, frame, cv::Size(3,3)); //medianBlur
}

void Pipeline::dilateErode(cv::Mat& frame)
{
	cv::dilate(frame, frame, _dilMat);
	cv::erode(frame, frame, _eroMat);
}

void Pipeline::draw2mat(std::vector < std::vector < cv::Rect > > &rect)
{
//	assert(!test.empty());
#if MY_DEBUG
	if (!trained[test].empty())
		cv::rectangle(_localFrame, trained[test][0], cv::Scalar(255, 0, 0), 3);
#endif
	for (uint j = 0; j < rect.size(); j++) {
		for (uint i = 0; i < rect[j].size(); i++) {
			cv::rectangle(_localFrame, rect[j][i], cv::Scalar(0, 255, 0), 3);
#if MY_DEBUG
			if (!trained[test].empty()) {//@DEBUG
		//		cv::rectangle(_localFrame, rect[j][i] & trained[test][0], cv::Scalar(255, 0, 0), 3);
			}
#endif
		}
	}
}

void Pipeline::draw2mat(std::vector < cv::Rect > &rect)
{
	for (uint i = 0; i < rect.size(); i++) {
		cv::rectangle(_localFrame, rect[i], cv::Scalar(0, 255, 0), 3);
#if MY_DEBUG
		if (!trained[test].empty()) {//@DEBUG
			cv::rectangle(_localFrame, rect[i] & trained[test][0], cv::Scalar(255, 0, 0), 3);
		}
#endif
	}
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
	if (ifs.is_open())	{
		getline(ifs, fileContents, '\x1A');
		ifs.close();
	}
	std::getline(ifs, fileContents, '\n');
	std::istringstream iss(fileContents);
	iss >> curLine;
	rects = std::vector< std::vector<cv::Rect> >(curLine);
	while (!iss.eof()) {
		cFrame = -1;
		iss >> cFrame >> x1 >> y1 >> x2 >> y2;
		cv::Point p1(x1, y1);
		cv::Point p2(x2, y2);
		if(cFrame >= 0)		rects[cFrame].push_back(cv::Rect(p1, p2));
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


void Pipeline::evaluate(std::map<std::string, int> & results)
{
	std::vector< std::vector<cv::Rect> > trained;
	std::vector< std::vector<cv::Rect> > test;
	loadRects(Settings::nameFile, test);
	loadRects(Settings::nameTrainedFile, trained);
	int truePos = 0, falsePos = 0, falseNeg = 0;
	std::vector< int > typeClass(test.size());

	for (int i = 0; i < test.size(); i++)
		falsePos += test[i].size();

	for (int i = 0; i < trained.size(); i++) {
		int isPedestrian = 0;
		for (int ii = 0; ii < trained[i].size(); ii++) {
			bool found = false;
			for (int jj = 0; jj < test[i].size(); jj++) {
				if (!trained[i].empty() && (trained[i][ii] & test[i][jj]).area() > (trained[i][ii].area() / 2)) {
					truePos++; // pedestrian found
					found = true;
					trained[i].erase(trained[i].begin() + ii);
					isPedestrian = 1;
				}
			}
			if (!found) falseNeg++;
			typeClass[i] = isPedestrian;
		}
	}
	falsePos -= truePos;

	float precision = static_cast<float>(truePos) / (truePos + falsePos); // Precision is the percentage true positives in the retrieved results.
	float recall = static_cast<float>(truePos) / (truePos + falseNeg); // Recall is the percentage of the pedestrians that the system retrieves.
	float f1score = 2 * (precision * recall) / (precision + recall);
	// ??% of the retrieved results were pedestrians, and ??% of the pedestrians were retrieved.

	std::cout << "True Positive: " << truePos << std::endl;
	std::cout << "False Positive: " << falsePos << std::endl;
	std::cout << "False Negative: " << falseNeg << std::endl;
	std::cout << "F1 score : " << f1score << " / " << f1score*100 << " %" << std::endl;
	//std::cout << "Precision : " << precision << std::endl;
	//std::cout << "Recall: " << recall << std::endl;
	std::cout << "END" << std::endl;
	
	results["tp"] = truePos;
	results["fp"] = falsePos;
	results["fn"] = falseNeg;
	results["f1"] = cvRound(f1score*100);

#if CALC_DIST
	std::vector< double > distances(test.size());
	for (uint i = 0; i < _rects2Eval.size(); i++) {
		for (uint j = 0; j < _rects2Eval[i].size(); j++) {
			for (uint k = 0; k < _rects2Eval[i][j].size(); k++) {
				if (!_distances.empty() && !_distances[i].empty() && !_distances[i][j].empty() ) {
					distances[i] = _distances[i][j][k];
				}
			}
		}
	}

	//distances.erase(
	//	std::remove(distances.begin(), distances.end(), 0),
	//	distances.end());
	//distances.shrink_to_fit();
	std::ofstream fs, fs2;
	fs.open(Settings::nameFile + "_distances.txt");
	fs2.open(Settings::nameFile + "_gt.txt");
	for (uint i = 0; i < distances.size(); i++) {
		if (distances[i] != 0) {
			fs << distances[i] << std::endl;
			fs2 << typeClass[i] << std::endl;
		}
	}
	fs.close();
	fs2.close();
#endif
}


