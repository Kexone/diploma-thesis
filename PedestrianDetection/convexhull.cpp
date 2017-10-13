#include "convexhull.h"
#include "settings.h"
#include "mainwindow.h"

ConvexHull::ConvexHull() {
    this->thresh = Settings::mogThreshold;
}

std::vector<std::vector<cv::Rect>> ConvexHull::wrapObjects(cv::Mat src, cv::Mat src_gray)
{
    cv::RNG rng(12345);
    cv::Mat threshold_output;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    /// Detect edges using Threshold
    assert(!src_gray.empty());
    threshold_output = src_gray.clone();
    //cv::threshold(src_gray, threshold_output, 180, 255, cv::THRESH_BINARY);
    /// Find contours
    assert(!threshold_output.empty());
    cv::findContours(src_gray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));

    /// Find the convex hull object for each contour
    std::vector<std::vector<cv::Point> >hull(contours.size());
    for (uint i = 0; i < contours.size(); i++)
    {
        convexHull(cv::Mat(contours[i]), hull[i], false);
    }

    std::vector<std::vector<cv::Point>>filteredHulls;
    int minThresholdArea = 5 * 100 , maxThresholdArea = 200 * 300; //max 400 * 400

    for (uint i = 0; i < hull.size(); i++) {
        int minX = src.cols, minY = src.rows, maxY = 0, maxX = 0;

        for (auto &p : hull[i]) {
            if (p.x <= minX) minX = p.x;
            if (p.y <= minY) minY = p.y;
            if (p.x >= maxX) maxX = p.x;
            if (p.y >= maxY) maxY = p.y;
        }

        // Vypočítej obsah
        if ((maxX - minX) * (maxY - minY) > minThresholdArea && (maxX - minX) * (maxY - minY) < maxThresholdArea)
             {
            filteredHulls.push_back(hull[i]);
        }
    }


    /// Draw contours + hull results
    std::vector<std::vector<cv::Rect>> react(filteredHulls.size());
   // cv::Mat drawing = cv::Mat::zeros(threshold_output.size(), CV_8UC3);
    for (uint i = 0; i < filteredHulls.size(); i++)
    {
        int minX = INT_MAX, minY = INT_MAX, maxY = 0, maxX = 0;
        for (auto &p : filteredHulls[i]) {
            if (p.x <= minX) minX = p.x;
            if (p.y <= minY) minY = p.y;
            if (p.x >= maxX) maxX = p.x;
            if (p.y >= maxY) maxY = p.y;
        }
        int size = 10;
        for(int s = 0; s < 0; s++) {
        if(minX >= 0 + size) minX -= size;
        if(minY >= 0 + size) minY -= size;
        if(maxX <= src.cols - size) maxX += size;
        if(maxY <= src.rows - size) maxY += size;
        size += 10;
        }
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//        //
//        std::stringstream ss;
//        ss << minX << " " << minY << " " << maxX << maxY;
//        std::string text = ss.str();
//        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SCRIPT_SIMPLEX,1, 2, 0);
//        cv::Point textOrg((src.cols - textSize.width)/2-150,(src.rows + textSize.height)/2);
//        cv::putText(src_gray,text,textOrg,cv::FONT_HERSHEY_SCRIPT_SIMPLEX,2,color,1);
//        //
        cv::drawContours(src_gray, contours, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        cv::drawContours(src_gray, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
       // drawContours(drawing, filteredHulls, i, color, 1, 8, std::vector<cv::Vec4i>(), 0, cv::Point());
        cv::Rect rectangle(cv::Point(minX, minY), cv::Point(maxX, maxY));
        if(rectangle.height / rectangle.width < 2) {
            int oldH = rectangle.height;
            rectangle.height = rectangle.width * 2;
            rectangle.y = rectangle.y -  ((rectangle.height - oldH) / 2);
        }
        react[i].push_back(rectangle);
    }

    /// Show in a window
    imshow("Hull demo", src_gray);
    threshold_output.release();
    return react;
}
