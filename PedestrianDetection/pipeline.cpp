#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::execute(std::vector<cv::Mat> frames)
{
    mog = Mog();
    hog = Hog();

    for(auto &frame: frames) {
        if(frame.empty()) {
            break;
        }
        process(frame);
    }
    cv::destroyWindow("Test");
}

void Pipeline::execute(int cameraFeed = 99)
{
    mog = Mog();
    hog = Hog();
    vs = new VideoStream(cameraFeed);
    vs->openCamera();
   // cv::namedWindow("Test",1);
    // TODO turn off alg
    for( ; ; ) {
        cv::Mat frame = vs->getFrame();
        if(frame.empty()) {
            break;
        }
        cv::imshow("Test", frame);
        //process(frame);
        frame.release();
    }
  //  cv::destroyWindow("Test");
}

void Pipeline::process(cv::Mat frame)
{
    localFrame = frame.clone();
    cv::Mat blured = frame.clone();
    cv::blur(frame, blured, cv::Size(10, 10));
    frame = mog.processMat(localFrame);
    cv::blur(frame, blured, cv::Size(10, 10));
    executeConvexHull(frame);

    std::vector<CroppedImage> croppedImages;
    if(rect.size() != 0) {
        cv::Mat croppedMat;
        for (uint j = 0; j < rect.size(); j++) {
            for (uint i = 0; i < rect[j].size(); i++) {
                croppedMat = frame.clone();
                croppedImages.emplace_back(CroppedImage(i,frame, rect[j][i]));
            }
        }
        croppedMat.release();
    }
    found_filtered = hog.detect(croppedImages);
    draw2mat(croppedImages);
    if(Settings::showVideoFrames)
        cv::imshow("Test", localFrame);
    cv::waitKey(20);
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
    ch = new ConvexHull(localFrame, frame, 0);
    rect = ch->thresh_callback(0, 0);
    delete ch;
}
