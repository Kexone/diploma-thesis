#include "fhog.h"
#include "../utils/utils.h"


Fhog::Fhog()
{
}
//dlib::image_window win;

Fhog::Fhog(std::string classPath)
{
	dlib::deserialize(classPath) >> detector;
}

void Fhog::detect(std::vector<CroppedImage>& frame, std::vector< std::vector < cv::Rect  > > &rects) try
{
	std::vector<std::vector<cv::Rect>> dets(frame.size());
	for (int i = 0; i < frame.size(); i++)
	{
		cv::Mat trz = frame[i].croppedImg;
		dlib::array2d<pixel_type> img;
		dlib::cv_image<pixel_type> temp(trz);
		dlib::assign_image(img, temp);

		dets[i] = Utils::vecDlibRectangle2VecOpenCV(detector(img, Settings::cropFhogAdjustTreshold), Settings::cropFhogMinArea);
	}

	rects = dets;
	dets.clear();
}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}
//dlib::image_window win;
void Fhog::detect(cv::Mat& frame, std::vector< cv::Rect > &rects) try
{
	dlib::array2d<pixel_type> img;
	dlib::cv_image<pixel_type> temp(frame);
	dlib::assign_image(img, temp);

	rects = Utils::vecDlibRectangle2VecOpenCV(detector(img, Settings::fhogAdjustTreshold), Settings::fhogMinArea);
}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}

float Fhog::predict(cv::Mat img, int flags)
{
	dlib::array2d < pixel_type > img1;
	dlib::cv_image<pixel_type> temp(img);
	dlib::assign_image(img1, temp);
	return 0;
}