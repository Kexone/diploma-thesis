#include "convexhull.h"

ConvexHull::ConvexHull() {
	this->_extensionTimes = Settings::cvxHullExtTimes;
	this->_extensionSize = Settings::cvxHullExtSize;
	this->_threshold = Settings::cvxHullThresh;
	this->_maxValue = Settings::cvxHullMaxValue;
}

void ConvexHull::wrapObjects(cv::Mat srcGray, std::vector< cv::Rect > &rects)
{
	assert(!srcGray.empty());
	
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
	
	_convexHullImage = srcGray.clone();

	cv::threshold(srcGray, _convexHullImage, _threshold, _maxValue, cv::THRESH_BINARY);
	//cv::threshold(srcGray, convexHullImage, 180, 255, cv::THRESH_BINARY);
    cv::findContours(_convexHullImage, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS, cv::Point(0, 0));

    /// Find the convex hull object for each contour
    std::vector<std::vector<cv::Point> > hulls( contours.size());
    for (uint i = 0; i < contours.size(); i++)	{
        cv::convexHull(cv::Mat(contours[i]), hulls[i], true);
    }

    std::vector <std::vector< cv::Point > > filteredHulls;
	filterByArea(hulls, filteredHulls);
	hulls.clear();

	rects = std::vector< cv::Rect > (filteredHulls.size());
	rects.clear();

    for (uint i = 0; i < filteredHulls.size(); i++)
    {
		cv::Rect rectangle = extendContours(filteredHulls[i]);
		rects.push_back(rectangle);
		//cv::RNG rng(12345);
      //  cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //    cv::drawContours(convexHullImage, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
      //  cv::drawContours(convexHullImage, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
    }
  //  imshow("Hull demo", convexHullImage);
  //  convexHullImage.release();

	if(rects.size() > 1)
	{
		clearInSameRegion(rects);
	}
}

void ConvexHull::filterByArea(std::vector<std::vector<cv::Point>>& hulls, std::vector<std::vector<cv::Point>>& filteredHulls)
{
	int minThresholdArea = 10 * 50, maxThresholdArea = 200 * 300; //max 400 * 400

	for (uint i = 0; i < hulls.size(); i++) {
		int minX = _convexHullImage.cols, minY = _convexHullImage.rows;
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
	int extS = _extensionSize;
	for (int s = 0; s < _extensionTimes; s++) {
		if (minX >= 0 + extS) minX -= extS;
		if (minY >= 0 + extS) minY -= extS;
		if (maxX <= _convexHullImage.cols - extS) maxX += extS;
		if (maxY <= _convexHullImage.rows - extS) maxY += extS;
		extS += _extensionSize;
	}
	return cv::Rect (cv::Point(minX, minY), cv::Point(maxX, maxY));
}

void ConvexHull::clearInSameRegion(std::vector<cv::Rect> &rects)
{
	int deviation = _extensionSize * _extensionTimes;
	for(uint i = 0; i < rects.size(); i++)
	{
		cv::Rect testRect = rects[i];
		for(uint t = i+1; t < rects.size(); t++)
		{
			if((testRect & rects[t] ).area() >= deviation)
			{
				rects.erase(rects.begin() + t,rects.end());
			}
		}
	}
}
