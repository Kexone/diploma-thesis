#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

class VideoStream
{
private:
    cv::VideoCapture capture;
    int camera = 99;
    std::string camSource;
public:
	static int fps;
	static int totalFrames;
	VideoStream();
    VideoStream(int cam);
    VideoStream(std::string camSource);
    cv::Mat getFrame();
    void openCamera();

};

#endif VIDEOSTREAM_H
