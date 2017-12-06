#include "fhog.h"
#include "../utils/utils.h"


FHog::FHog()
{
}

std::vector< std::vector < cv::Rect  > > FHog::detect(std::vector<CroppedImage>& frame)
{
dlib::image_window win;
	std::vector< std::vector < cv::Rect  > > dets (frame.size()) ;
	typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
	image_scanner_type scanner;
	//scanner.set_detection_window_size(80, 80);

	for (int i = 0; i < frame.size(); i++) {
		cv::Mat trz = frame[0].croppedImg;
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::bgr_pixel> temp(trz);
		dlib::assign_image(img, temp);


		dlib::object_detector<image_scanner_type> detector;
		dlib::deserialize("dlib_pedestrian_detector.svm") >> detector;

		if(detector(img).empty())
			continue;

		 dets[i] = (Utils::vecDlibRectangle2VecOpenCV(detector(img)));
		 dlib::image_window win;
		 win.clear_overlay();
		 win.set_image(img);
		 win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));
	}
	
	//std::cout << "Hit enter to process the next image..." << std::endl;
	//std::cin.get();

	return dets;
}
