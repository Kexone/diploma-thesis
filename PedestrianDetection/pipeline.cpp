#include "pipeline.h"
#include <sstream>
Pipeline::Pipeline()
{

}


int Pipeline::execute(std::vector<cv::Mat> frames)
{
    set();

    for(auto &frame: frames) {
        if(frame.empty()) {
            break;
        }
       process(frame);
       frame.release();
    }
    cv::destroyWindow("Result");
    return allDetections;
}

int Pipeline::execute(int cameraFeed = 99)
{
    set();
    vs = new VideoStream(cameraFeed);
    vs->openCamera();
   // cv::namedWindow("Test",1);
    // TODO turn off alg
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty() || interrupt) {
            break;
        }
        debugMog(frame);
        //process(frame);
        frame.release();
    }
  //  cv::destroyWindow("Test");
    return allDetections;

}

int Pipeline::execute(std::string cameraFeed)
{
    set();
    vs = new VideoStream(cameraFeed);
    vs->openCamera();
    int i = 0;
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty() || interrupt) {
            break;
            delete vs;
        }
        debugMog(frame);
        //process(frame);
        frame.release();
        std::stringstream ss;
        ss <<  "../img/mat_" << i << ".jpg";
        cv::imwrite(ss.str(),localFrame);
        localFrame.release();
        i++;
    }
     cv::destroyWindow("Test");
     return allDetections;
}

void Pipeline::process(cv::Mat frame)
{
    localFrame = frame.clone();
    cv::Mat blured;
    cv::blur(frame, blured, cv::Size(5, 5));
    cv::cvtColor(blured,blured, CV_BGR2GRAY);
    frame = mog.processMat(blured);
    cv::imshow("mog", frame);
   // cv::blur(frame, blured, cv::Size(5, 5));
    cv::Canny(frame, frame, 10, 130, 3);
    rect = ch.wrapObjects(localFrame, frame);
    std::vector<CroppedImage> croppedImages;
    if(rect.size() != 0) {
        for (uint j = 0; j < rect.size(); j++) {
            for (uint i = 0; i < rect[j].size(); i++) {
                croppedImages.emplace_back(CroppedImage(i,localFrame.clone(), rect[j][i]));
            }
        }
    }
    found_filtered = hog.detect(croppedImages);
    draw2mat(croppedImages);
    if(Settings::showVideoFrames)
    cv::imshow("Result", localFrame);
    cv::waitKey(5);
    frame.release();
    blured.release();
}

void Pipeline::draw2mat(std::vector<CroppedImage> croppedImages)
{
    for (uint j = 0; j < found_filtered.size(); j++) {
        for (uint i = 0; i < found_filtered[j].size(); i++) {
            cv::Rect r = found_filtered[j][i];
            r.x += cvRound(croppedImages[j].offsetX);
            //r.width = cvRound(croppedImages[j].croppedImg.cols);
            r.y += cvRound(croppedImages[j].offsetY);
            //r.height = cvRound(croppedImages[j].croppedImg.rows);
            cv::rectangle(localFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
            allDetections += found_filtered[j].size();
        }
    }
}

void Pipeline::set()
{
    interrupt = false;
    allDetections = 0;
    mog = Mog();
    hog = Hog();
    ch = ConvexHull();
}

void Pipeline::interruptDetection()
{
    this->interrupt = true;
}

void Pipeline::debugMog(cv::Mat frame)
{
    char k;
    localFrame = frame.clone();
    cv::cvtColor(frame,frame,CV_BGR2GRAY);
    cv::imshow("Blur", frame);
    cv::blur(frame, frame, cv::Size(6,6));
//    cv::threshold(frame, frame, 90, 255, cv::THRESH_BINARY);
   // cv::imshow("Thresh", frame);
   // cv::Sobel(frame,frame,0,0,1,5,3,0,cv::BORDER_DEFAULT);
   // cv::Scharr(frame,frame,0,0,1,5,0, cv::BORDER_DEFAULT);
    //cv::Canny(frame, frame, 25, 40, 5);
    cv::imshow("canny", frame);
    frame = mog.processMat(frame);
    cv::blur(frame, frame, cv::Size(9,9));
    //cv::threshold(frame, frame, 90, 255, cv::THRESH_BINARY);
    cv::imshow("MOG", frame);
    debugCHHOG(frame);
    cv::waitKey(5);
   // k=cvWaitKey(0);
    frame.release();
}

void Pipeline::debugCHHOG(cv::Mat frame)
{
    rect = ch.wrapObjects(localFrame, frame);

    std::vector<CroppedImage> croppedImages;
    if(rect.size() != 0) {
        for (uint j = 0; j < rect.size(); j++) {
            for (uint i = 0; i < rect[j].size(); i++) {
                croppedImages.emplace_back(CroppedImage(i,localFrame.clone(), rect[j][i]));
            }
        }
    }
    found_filtered = hog.detect(croppedImages);
    draw2mat(croppedImages);
    if(Settings::showVideoFrames)
    cv::imshow("Result", localFrame);
}
