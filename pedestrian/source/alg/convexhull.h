#ifndef CONVEXHULL_H
#define CONVEXHULL_H
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

class ConvexHull
{
public:
    ConvexHull();

	/**  
	* @brief Finds contours in cropped binary image from the original frame.
	* Method is threaded by filtering sizes and  same position of contours.
	*
	* @param srcGray binary frame after background substraction (default used MoG2)
	* @return vector of cv::Rects
	*/
    std::vector< cv::Rect > wrapObjects(cv::Mat srcGray);

private:
	cv::Mat convexHullImage;
    double thresh;
	int extensionSize;
	int extensionTimes;

	/**
	 * @brief Filters contours by size of itself.
	 * The unsatisfactory contours be thrown away. Filtering by threshold area.
	 * 
	 * @param hulls 
	 * @param filteredHulls
	 */
	void filterByArea(std::vector< std::vector< cv::Point > > &hulls, std::vector< std::vector< cv::Point > > &filteredHulls);

	/**
	 * @brief Extends the contours in image about 10 pixels four-times
	 * 
	 * @param hull vector of cv::Points to resize
	 * @return extendend rectangles
	 */
	cv::Rect extendContours(std::vector< cv::Point > &hull);

	/**
	 * @brief Destroys the contours in the same region and save only the first
	 * 
	 * @param rects vector of cv::Rect 
	 */
	void clearInSameRegion(std::vector< cv::Rect > &rects);
};

#endif // CONVEXHULL_H


