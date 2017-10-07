#include "pipeline.h"

Pipeline::Pipeline()
{

}

void Pipeline::chooseType(int type, std::vector<cv::Mat> frames)
{
    //cv::Mat test;x
    int tpe = type;
    mog = Mog();
    hog = Hog();
    std::vector<std::vector<cv::Rect>> found_filtered;
    std::vector<std::vector<cv::Rect>> rect;

    for(auto &frame: frames) {
        if(frame.empty()) {
            cv::destroyWindow("Test");
            return;
        }
        cv::Mat src_orig = frame.clone();
        cv::Mat blured = frame.clone();
        cv::blur(frame, blured, cv::Size(10, 10));
        frame = mog.processMat(src_orig);
        cv::blur(frame, blured, cv::Size(10, 10));
        ch = new ConvexHull(src_orig, frame, 0);
        rect = ch->thresh_callback(0, 0);
        delete ch;
        std::vector<CroppedImage> croppedImages;
        if(rect.size() != 0) {
            cv::Mat croppedMat;
            for (uint j = 0; j < rect.size(); j++) {
                for (uint i = 0; i < rect[j].size(); i++)
                {
                    croppedMat = frame.clone();
                    croppedImages.emplace_back(CroppedImage(i,frame, rect[j][i]));
                }
            }
            croppedMat.release();
        }
        found_filtered = hog.detect(croppedImages);
        for (uint j = 0; j < found_filtered.size(); j++) {
            for (uint i = 0; i < found_filtered[j].size(); i++) {
                cv::Rect r = found_filtered[j][i];
                r.x += cvRound(croppedImages[j].offsetX);
                r.width = cvRound(croppedImages[j].croppedImg.cols);
                r.y += cvRound(croppedImages[j].offsetY);
                r.height = cvRound(croppedImages[j].croppedImg.rows);
                rectangle(src_orig, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
            }
        }
      //  cv::imshow("Test", src_orig);
        cv::waitKey(20);
        frame.release();
    }
    cv::destroyWindow("Test");
}
