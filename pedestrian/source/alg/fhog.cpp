#include "fhog.h"
#include "../utils/utils.h"


FHog::FHog()
{
}

FHog::FHog(std::string classPath)
{
//	dlib::deserialize("dlib_pedestrian_detector.svm") >> detector;
//	dlib::deserialize(classPath) >> detector;
	dlib::deserialize("D://Codes//school//backupDT//repo//pedestrian//pedDet.svm") >> detector;
}

void FHog::detect(std::vector<CroppedImage>& frame, std::vector< std::vector < cv::Rect  > > &rects) try
{
//dlib::image_window win;
	std::vector< std::vector < cv::Rect  > > dets (frame.size()) ;
	//scanner.set_detection_window_size(80, 80);
	dlib::deserialize("pedDet.svm") >> detector;


	for (int i = 0; i < frame.size(); i++) {
		cv::Mat trz = frame[i].croppedImg;
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::bgr_pixel> temp(trz);
		dlib::assign_image(img, temp);

		if(detector(img).empty())
			continue;

		 dets[i] = (Utils::vecDlibRectangle2VecOpenCV(detector(img)));
		 //dlib::image_window win;
		// win.clear_overlay();
		// win.set_image(img);
	//	 win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));
	}

	rects = dets;
	dets.clear();
}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}
//dlib::image_window win;
void FHog::detect(cv::Mat& frame, std::vector< cv::Rect > &rects)// try
{
	//dlib::image_window win;
	//dlib::deserialize("D://Codes//school//backupDT//repo//pedestrian//pedDet.svm") >> detector;
	//scanner.set_detection_window_size(80, 80);
//		auto startTime = std::chrono::high_resolution_clock::now();
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::rgb_pixel> temp(frame);
		dlib::assign_image(img, temp);
//		auto endTime = std::chrono::high_resolution_clock::now();
//		double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
//		std::cout << "CONVERT " << time / 1000 << std::endl;
	//cv::imshow("t", frame);
	//cv::waitKey(0);
//	dlib::cv_image<dlib::bgr_pixel> image(frame);
//	dlib::matrix<dlib::rgb_pixel> matrix;
//		assign_image(matrix, image);
	//	if (detector(img,1).empty())
		//	return;
//		startTime = std::chrono::high_resolution_clock::now();
		rects = (Utils::vecDlibRectangle2VecOpenCV(detector(img,0.8)));
//		endTime = std::chrono::high_resolution_clock::now();
//		time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
	//	std::cout << "DETECT " << time / 1000 << std::endl;
		//win.clear_overlay();
		//win.set_image(img);
	//	win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));

}
//catch (std::exception& e)
//{
	//std::cout << e.what() << std::endl;
//}