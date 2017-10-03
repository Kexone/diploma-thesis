#include "hog.h"

Hog::Hog()
{
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

std::vector<std::vector<cv::Rect> > Hog::detect(std::vector<cv::Mat> frames)
{
    std::vector<std::vector<cv::Rect>> found_filtered(frames.size());
    if (frames.empty())
        return found_filtered;
    fflush(stdout);
    for (int x = 0; x < frames.size(); x++) {
        std::vector<cv::Rect> rRect;
        std::vector<cv::Rect> found;
        //cv::Mat test  = frames[x].croppedImg;
        //cv::Size size(64, 128);
        //hog.detectMultiScale(test, found, 0, cv::Size(6, 6), cv::Size(32, 32), 1.05, 2);
        hog.detectMultiScale(frames.at(x), found, 0, cv::Size(4, 4), cv::Size(32,32), 1.05, 2);
        if (found.empty()) {
            continue;
        }
        size_t i, j;
        for (i = 0; i<found.size(); i++)
        {
            cv::Rect r = found[i];
            for (j = 0; j<found.size(); j++)
                if (j != i && (r & found[j]) == r)
                    break;
                if (j == found.size())
                    found_filtered[x].push_back(r);
        }
        found.clear();
    }
    return found_filtered;
}
