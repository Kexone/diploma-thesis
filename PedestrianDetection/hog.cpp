#include "hog.h"


Hog::Hog()
{
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

std::vector<cv::Rect> Hog::detect(cv::Mat frame)
{
    std::vector<cv::Rect> found_filtered;
    cv::Mat src_frame = frame.clone();
    if (frame.empty())
        return found_filtered;
    fflush(stdout);
    //for (uint x = 0; x < frames.size(); x++) {
        std::vector<cv::Rect> rRect;
        std::vector<cv::Rect> found;
        //cv::Mat test  = frames[x].croppedImg;
        //cv::Size size(64, 128);
        cv::resize(src_frame, src_frame, cv::Size(64,128));
        //hog.detectMultiScale(test, found, 0, cv::Size(6, 6), cv::Size(32, 32), 1.05, 2);
        hog.detectMultiScale(src_frame, found, 0, cv::Size(8, 8), cv::Size(32,32), 1.05, 0);
        if (found.empty()) {
            return found_filtered;
        }
        size_t i, j;
        for (i = 0; i<found.size(); i++)
        {
            cv::Rect r = found[i];
            for (j = 0; j<found.size(); j++)
                if (j != i && (r & found[j]) == r)
                    break;
                if (j == found.size())
                    found_filtered.push_back(r);
        }
        found.clear();
    //}
    return found_filtered;
}

std::vector<std::vector<cv::Rect>> Hog::detect(std::vector<CroppedImage>& frames) {

    std::vector<std::vector<cv::Rect>> found_filtered(frames.size());
    if (frames.empty())
        return found_filtered;
        fflush(stdout);
        for (uint x = 0; x < frames.size(); x++) {
            std::vector<cv::Rect> rRect;
            std::vector<cv::Rect> found;
            cv::Mat test  = frames[x].croppedImg;
            assert(!test.empty());
            cv::resize(test, test, cv::Size(64,128));
            cv::imshow("test", test);
            test.convertTo(test,CV_8UC1);
            //cv::Mat test = resizeImage(frames[x].croppedImg, cv::Size(64, 128));
           // hog.blockStride = cv::Size(test.cols/4, test.rows/8);
            //hog.blockSize = cv::Size(test.cols/4, test.rows/8);
            hog.winSize = cv::Size(test.cols, test.rows);
            hog.detectMultiScale(test, found, 0, cv::Size(4,4), cv::Size(16,16), 1.0, 2);
          //  hog.detectMultiScale(test, found, 0, cv::Size(4, 4), cv::Size(32,32), 1.05, 2);

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
