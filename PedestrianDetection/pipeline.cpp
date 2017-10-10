#include "pipeline.h"

Pipeline::Pipeline()
{

}

Pipeline::~Pipeline()
{
    delete vs;
}

void Pipeline::execute(std::vector<cv::Mat> frames)
{
    mog = Mog();
    hog = Hog();

    for(auto &frame: frames) {
        if(frame.empty()) {
            break;
        }
        //debug
        //cv::Mat blured;
        //cv::blur(frame, blured, cv::Size(5, 5));
        //frame = mog.processMat(blured);
        //cv::blur(frame, localFrame, cv::Size(10, 10));
        //cv::imshow("MOG-test", frame);
//        //

       process(frame);

        //debug
       // cv::waitKey(5);
        //frame.release();
        //blured.release();
        //
    }
    cv::destroyWindow("Result");
}

void Pipeline::execute(int cameraFeed = 99)
{
    interrupt = false;
    mog = Mog();
    hog = Hog();
    vs = new VideoStream(cameraFeed);
    vs->openCamera();
   // cv::namedWindow("Test",1);
    // TODO turn off alg
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty() || interrupt) {
            break;
        }
        cv::blur(frame, frame, cv::Size(5, 5));
       // cv::Sobel(blured, blured,3,0,0,3,1,0);
        frame = mog.processMat(frame);
        cv::imshow("Test", frame);
        //process(frame);
        cv::waitKey(5);
        frame.release();
    }
  //  cv::destroyWindow("Test");
}

void Pipeline::execute(std::string cameraFeed)
{
    this->interrupt = false;
    mog = Mog();
    hog = Hog();
    vs = new VideoStream(cameraFeed);
    vs->openCamera();

   // cv::namedWindow("Test",1);
    // TODO turn off alg
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty() || interrupt) {
            vs->closeCamera();
            delete vs;
            break;
        }
        debugMog(frame);
        //process(frame);
        cv::waitKey(5);
        frame.release();
    }
     cv::destroyWindow("Test");
}

void Pipeline::process(cv::Mat frame)
{
    localFrame = frame.clone();
    cv::Mat blured;
    cv::blur(frame, blured, cv::Size(5, 5));
//    blured.convertTo(blured, CV_32SC1);
    cv::cvtColor(blured,blured, CV_BGR2GRAY);
    frame = mog.processMat(blured);
    cv::imshow("mog", frame);
   // cv::blur(frame, blured, cv::Size(5, 5));
    executeConvexHull(frame);

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
            r.width = cvRound(croppedImages[j].croppedImg.cols);
            r.y += cvRound(croppedImages[j].offsetY);
            r.height = cvRound(croppedImages[j].croppedImg.rows);
            cv::rectangle(localFrame, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
        }
    }
}

void Pipeline::executeConvexHull(cv::Mat frame)
{
    ch = ConvexHull(localFrame, frame);
    rect = ch.wrapObjects(localFrame, frame);
    //delete ch;
}

void Pipeline::interruptDetection()
{
    this->interrupt = true;
}

void Pipeline::debugMog(cv::Mat frame)
{
    cv::blur(frame, frame, cv::Size(5, 5));
   // cv::Sobel(blured, blured,3,0,0,3,1,0);
    frame = mog.processMat(frame);
    cv::Canny(frame, frame, 10, 130, 3);
    cv::imshow("Test", frame);
    cv::waitKey(5);
    frame.release();
}
