#ifndef VIDEOSTREAM_H
#define VIDEOSTREAM_H

#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

/**
 * class VideoStream
 */
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
	/**
	* @brief Get frame from stream.
	*
	* @return next frame from stream
	*/
    cv::Mat getFrame();
	/**
	* @brief Opens videostream, it's depends on @camera or @camSource if opens videostream from camera or from file.
	*/
    void openCamera();

};

#endif // VIDEOSTREAM_H
