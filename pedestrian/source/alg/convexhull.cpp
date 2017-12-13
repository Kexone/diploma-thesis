#include "convexhull.h"
#include "../settings.h"

ConvexHull::ConvexHull() {
	this->thresh = 115; // Settings::mogThreshold;
	this->extensionTimes = 4;
	this->extensionSize = 10;
}

// @TODO refactor this class 
std::vector<cv::Rect> ConvexHull::wrapObjects(cv::Mat src, cv::Mat srcGray)
{
	assert(!srcGray.empty());
	
	cv::RNG rng(12345);
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
	
	convexHullImage = srcGray.clone();
    cv::threshold(srcGray, convexHullImage, 180, 255, cv::THRESH_BINARY);
    /// Find contours
    cv::findContours(convexHullImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS, cv::Point(0, 0));

    /// Find the convex hull object for each contour
    std::vector<std::vector<cv::Point> > hulls( contours.size());
    for (uint i = 0; i < contours.size(); i++)
    {
        convexHull(cv::Mat(contours[i]), hulls[i], true);
    }

    std::vector <std::vector< cv::Point > > filteredHulls;
	filterByArea(hulls, filteredHulls);


    std::vector< cv::Rect > rects (filteredHulls.size());
	rects.clear();
    for (uint i = 0; i < filteredHulls.size(); i++)
    {
		cv::Rect rectangle = extendContours(filteredHulls[i]);

        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::drawContours(convexHullImage, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        cv::drawContours(convexHullImage, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());

        rects.push_back(rectangle);
    }
    /// Show in a window
    imshow("Hull demo", convexHullImage);

	convexHullImage.release();

	//std::cout << " CH size: " << rects.size() << std::endl;
	if(rects.size() > 1)
	{
		//std::cout << "		CH1 size: " << rects[0].size() << std::endl;
		//		std::cout << "		CH2 size: " << rects[1].size() << std::endl;

		clearInSameRegion(rects);
	}
    return rects;
}

void ConvexHull::filterByArea(std::vector<std::vector<cv::Point>>& hulls, std::vector<std::vector<cv::Point>>& filteredHulls)
{
	int minThresholdArea = 10 * 50, maxThresholdArea = 200 * 300; //max 400 * 400

	for (uint i = 0; i < hulls.size(); i++) {
		int minX = convexHullImage.cols, minY = convexHullImage.rows;
		int maxY = 0, maxX = 0;

		for (auto &p : hulls[i]) {
			if (p.x <= minX) minX = p.x;
			if (p.y <= minY) minY = p.y;
			if (p.x >= maxX) maxX = p.x;
			if (p.y >= maxY) maxY = p.y;
		}

		// Calc area
		if ((maxX - minX) * (maxY - minY) > minThresholdArea && (maxX - minX) * (maxY - minY) < maxThresholdArea)
		{
			filteredHulls.push_back(hulls[i]);
		}
	}
	hulls.clear();
}

cv::Rect ConvexHull::extendContours(std::vector<cv::Point>& hull)
{
	int minX = INT_MAX, minY = INT_MAX, maxY = 0, maxX = 0;
	for (auto &p : hull) {
		if (p.x <= minX) minX = p.x;
		if (p.y <= minY) minY = p.y;
		if (p.x >= maxX) maxX = p.x;
		if (p.y >= maxY) maxY = p.y;
	}
	int extS = extensionSize;
	for (int s = 0; s < extensionTimes; s++) {
		if (minX >= 0 + extS) minX -= extS;
		if (minY >= 0 + extS) minY -= extS;
		if (maxX <= convexHullImage.cols - extS) maxX += extS;
		if (maxY <= convexHullImage.rows - extS) maxY += extS;
		extS += extensionSize;
	}
	return cv::Rect (cv::Point(minX, minY), cv::Point(maxX, maxY));
}

void ConvexHull::clearInSameRegion(std::vector<cv::Rect> &rects)
{
	int deviation = extensionSize * extensionTimes;
	for(uint i = 0; i < rects.size(); i++)
	{
		cv::Rect testRect = rects[i];
		for(uint t = i+1; t < rects.size(); t++)
		{
		//	std::cout << testRect.x - rects[t].x << std::endl;
			if(testRect.x - rects[t].x < deviation)
			{
				rects.erase(rects.begin() + t,rects.end());
			}
		}
	
	}
}
