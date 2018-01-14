#ifndef EXTRACTORROI_H
#define EXTRACTORROI_H
#include "../media/videostream.h"
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/videostab/ring_buffer.hpp>

//#include <windows.h>

/**
 * class ExtractorROI
 * 
 * This class serve as extractor ROI from frame 
 * Extracting is going by mouse drag rectangle to frame and save coordinate to file
 */
class ExtractorROI
{
private: 
	bool clicked = false;
	VideoStream *vs;
	cv::Mat fullFrame;
	cv::Mat img;
	cv::Rect cropRect;
	cv::Point point1;
	cv::Point point2;
	std::string path;
	const std::string WIN_NAME = "EXTRACT ROI";
	char imgName[15];
	const int N_RECT;
	int indRect;
	std::vector < cv::Mat > ROIs;
	std::vector < cv::Rect > rects;
	std::vector < std::vector  < cv::Rect > > rects2Save;

public:
	/**
	 * @constructor ExtractorROI
	 * 
	 * @param rectCount count of all rects what will be used to annotate pedestrians
	 */
	ExtractorROI(int rectCount) : N_RECT(rectCount) {};

	/**
	 * @brief Annotate is working only on frames from video stream
	 * This method firstly create folder with name of video and ROI folder, inside this folder
	 * Sets size of vector to number of all frames from used videostream
	 * 
	 * @param videoStreamPath path to video stream
	 */
	void extractROI(std::string videoStreamPath);

private:

	/**
	 * @brief Writes the vector of annotated rectangles into file
	 * First line is numbers of frames from video stream, on next lines are written number of current frame and points of rectangles
	 * Every rectangles is written on line separately
	 */
	void write2File();
	
	/**
	 * @brief Core of the class. Is called from public method in cycle
	 * Captures keys to process image, changing size and position of rectangle
	 * save frame and ROI and switching between rectangles if are available
	 * Sets the mouse callback on window which allow the draw rectangle
	 * 
	 * @param cFrame number of actual frame
	 */
	void process(int cFrame);

	/**
	 * @brief Show frame with rectangle (if rectangle available)
	 */
	void showImage();

	/**
	 * @brief Draws the rectangles into image, but first check boundaries
	 * Shows the ROI images and destroys windows if their rectangle is equals zero
	 */
	void drawRects();

	/**
	 * @brief Checks if rectangle is outside of image and shrinkes it
	 */
	void checkBoundary();

	/**
	 * @brief Mouse callback to allows draw into image
	 * 
	 * @param event
	 * @param x mouse coord
	 * @param y mouse coord
	 */
	void onMouse(int event, int x, int y);

	/**
	* @brief Wrapper for mouse callback
	*
	* @param event
	* @param x mouse coord
	* @param y mouse coord
	* @param userData
	*/
	static void onMouse(int event, int x, int y, int, void* userdata);

	
};

#endif //EXTRACTORROI_H