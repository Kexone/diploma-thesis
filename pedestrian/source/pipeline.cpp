#include "pipeline.h"
#include <sstream>

#define PURE_HOG 1
#define MIXTURED_HOG 2
#define PURE_FHOG 3
#define MIXTURED_FHOG 4
#define PURE_CASCADE 5
#define MIXTURED_CASCADE 6

int Pipeline::allDetections = 0;

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
		_fhog = new FHog("data.svm");
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
	allDetections = 0;
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
	cv::namedWindow("Result", CV_WINDOW_AUTOSIZE);
	while (sampleFile >> oSample) {
		frame = cv::imread(oSample, CV_LOAD_IMAGE_UNCHANGED);
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

		cv::imshow("Result", _localFrame);
		cv::waitKey(2);
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
	_localFrame = frame.clone();
	preprocessing(frame);
	//auto startTime = std::chrono::high_resolution_clock::now();
	_mog.processMat(frame);
	//auto endTime = std::chrono::high_resolution_clock::now();
	//double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
	//std::cout << "MOG took " << static_cast<float>(time) / CLOCKS_PER_SEC << "s." << std::endl;
	dilateErode(frame);

	std::vector< cv::Rect > rect;
	//startTime = std::chrono::high_resolution_clock::now();
	_ch.wrapObjects(frame, rect);
	//endTime = std::chrono::high_resolution_clock::now();
	//time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
	//std::cout << "CH  took " << static_cast<float>(time) / CLOCKS_PER_SEC << "s." << std::endl;
	//cv::imshow("mog", frame);
	if (rect.size() != 0) {
		std::vector< CroppedImage > croppedImages;
		for (size_t i = 0; i < rect.size(); i++) {
			croppedImages.push_back(CroppedImage(i, _localFrame.clone(), rect[i]));
		}
		std::vector < std::vector < cv::Rect > > foundRect;
		std::vector < std::vector < float > > distances(croppedImages.size());

		//startTime = std::chrono::high_resolution_clock::now();
		_hog.detect(croppedImages, foundRect, distances);
		//endTime = std::chrono::high_resolution_clock::now();
		//time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
		//std::cout << "HOG took " << static_cast<float>(time) / CLOCKS_PER_SEC << "s." << std::endl;
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
	//preprocessing(frame);

	std::vector < cv::Rect > foundRect;

	_hog.detect(frame, foundRect);
	draw2mat(foundRect);
	if(!foundRect.empty())
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
	//preprocessing(frame);

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
	std::vector < cv::Rect  > foundRect;
	_hog.detect(frame, foundRect); 
	draw2mat(foundRect);

	if(Settings::showVideoFrames)
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

void Pipeline::draw2mat(std::vector< CroppedImage > &croppedImages, std::vector < std::vector < cv::Rect > > &rect)
{
	for (uint j = 0; j < rect.size(); j++) {
		for (uint i = 0; i < rect[j].size(); i++) {
			cv::rectangle(_localFrame, rect[j][i], cv::Scalar(0, 255, 0), 3);
#if MY_DEBUG
			if (!trained[test].empty()) {//@DEBUG
				cv::rectangle(_localFrame, rect[j][i] & trained[test][0], cv::Scalar(255, 0, 0), 3);
			}
#endif
		}
		allDetections += rect[j].size();
	}
	rect.clear();
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

	for (int i = 0; i < test.size(); i++)
		falsePos += test[i].size();

	for (int i = 0; i < trained.size(); i++) {
	//	for (int j = 0; j < test.size(); j++) {
	//		if (i == j)
				for (int ii = 0; ii < trained[i].size(); ii++) {
					bool found = false;
					for (int jj = 0; jj < test[i].size(); jj++) {
						if (!trained[i].empty() && (trained[i][ii] & test[i][jj]).area() >(trained[i][ii].area() / 2)) {
							truePos++; // pedestrian found
							found = true;
							trained[i].erase(trained[i].begin() + ii);
						}
					}
					if (!found) falseNeg++;
				}
//		}
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
}


