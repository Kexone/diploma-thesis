#ifndef EXTRACTORROI_H
#define EXTRACTORROI_H
#include "../media/videostream.h"
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>

/*
 * This class serve as extractor ROI from frame 
 * Extracting is going by mouse drag rectangle to frame and save coordinate to file
 */
class ExtractorROI
{
private: 
	VideoStream *vs;
	std::string nameFile;
	bool clicked = false;
	cv::Mat fullFrame;
	cv::Mat img;
	cv::Mat ROI;
	cv::Rect cropRect;
	cv::Point point1;
	cv::Point point2;
	char imgName[15];
	int rectCount;
	int indRect;
	std::vector < cv::Rect > rects;
	std::vector < std::vector  < cv::Rect > > rects2Save;
	const std::string winName = "EXTRACT ROI";

public:
	ExtractorROI(int rectCount, std::string nameOfFile) : rectCount(rectCount), nameFile(nameOfFile) {};
	void extractROI(std::string cameraFeed);

private:
	void write2File();
	void process(int cFrame);
	void showImage();
	void drawRects();
	void checkBoundary();
	void onMouse(int event, int x, int y);
	static void onMouse(int event, int x, int y, int, void* userdata);

	
};

#endif //EXTRACTORROI_H