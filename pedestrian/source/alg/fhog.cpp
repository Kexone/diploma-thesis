#include "fhog.h"
#include "../utils/utils.h"


FHog::FHog()
{
}

FHog::FHog(std::string classPath)
{
//	dlib::deserialize("dlib_pedestrian_detector.svm") >> detector;
	dlib::deserialize(classPath) >> detector;

}

void FHog::detect(std::vector<CroppedImage>& frame, std::vector< std::vector < cv::Rect  > > &rects)
{
dlib::image_window win;
	std::vector< std::vector < cv::Rect  > > dets (frame.size()) ;
	//scanner.set_detection_window_size(80, 80);

	for (int i = 0; i < frame.size(); i++) {
		cv::Mat trz = frame[0].croppedImg;
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::bgr_pixel> temp(trz);
		dlib::assign_image(img, temp);

		if(detector(img).empty())
			continue;

		 dets[i] = (Utils::vecDlibRectangle2VecOpenCV(detector(img)));
		 dlib::image_window win;
		 win.clear_overlay();
		 win.set_image(img);
		 win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));
	}

	rects = dets;
	dets.clear();
}

void FHog::detect(cv::Mat& frame, std::vector< cv::Rect > &rects)
{
	dlib::image_window win;
	
	//scanner.set_detection_window_size(80, 80);

		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::bgr_pixel> temp(frame);
		dlib::assign_image(img, temp);

		if (detector(img).empty())
			return;

		rects = (Utils::vecDlibRectangle2VecOpenCV(detector(img)));
//		win.clear_overlay();
//		win.set_image(img);
//		win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));

}