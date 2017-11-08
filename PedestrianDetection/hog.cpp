#include "hog.h"
#include <QDebug>

Hog::Hog()
{
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("96_16_8_8_9_01.yml");
    std::vector< float > hogDetector;
    getSvmDetector(svm,hogDetector);
    hog.setSVMDetector(hogDetector);
    //hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
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
        //cv::resize(src_frame, src_frame, cv::Size(64,128));
        //hog.detectMultiScale(test, found, 0, cv::Size(6, 6), cv::Size(32, 32), 1.05, 2);
        hog.detectMultiScale(src_frame, found, 1, cv::Size(8, 8), cv::Size(32,32), 1.05, 0);
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
            //cv::resize(test, test, cv::Size(64,128));

            test.convertTo(test,CV_8UC1);
            //cv::Mat test = resizeImage(frames[x].croppedImg, cv::Size(64, 128));
           // hog.blockStride = cv::Size(test.cols/4, test.rows/8);
            //hog.blockSize = cv::Size(test.cols/4, test.rows/8);
            //hog.winSize = cv::Size(test.cols, test.rows);
            hog.detectMultiScale(test, found, 0, cv::Size(8,8), cv::Size(0,0), 1.05, 2);
          //  hog.detectMultiScale(test, found, 0, cv::Size(4, 4), cv::Size(32,32), 1.05, 2);

            if (found.empty()) {
                continue;
            }
            size_t i, j;
            for (i = 0; i<found.size(); i++)
            {
                cv::Rect r = found[i];

                //for (j = 0; j<found.size(); j++)
                //    if (j != i && (r & found[j]) == r)
                //        break;
                //if (j == found.size())
                    found_filtered[x].push_back(r);
                    qDebug() << "TL" << found[i].tl().x << found[i].tl().y << " BR" << found[i].br().x << found[i].br().y;
                    cv::rectangle(test, found[i].tl(), found[i].br(),cv::Scalar(0,0,255),4,8,0);
            }
            cv::imshow("test", test);
            //found.clear();
        }
        return found_filtered;
}

void Hog::getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector )
{
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    //CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    //CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
      //         (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    //CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
}
