#include "myFhog.h"
#include "../utils/utils.h"


MyFHog::MyFHog()
{
}
//dlib::image_window win;

MyFHog::MyFHog(std::string classPath)
{
//	dlib::deserialize("dlib_pedestrian_detector.svm") >> detector;
//	dlib::deserialize(classPath) >> detector;
	//dlib::deserialize("D://Codes//school//backupDT//repo//pedestrian//pedDet.svm") >> detector;
	//scanner.set_detection_window_size(80, 80);
	//scanner.set_max_pyramid_levels(1);

	//detector = dlib::object_detector<image_scanner_type>(scanner, detector.get_overlap_tester(), detector.get_w());
	//dlib::deserialize("pedDet.svm") >> scanner;
	//detector.scanner = scanner;
	dlib::deserialize("trained.svm") >> detector;


}

void MyFHog::detect(std::vector<CroppedImage>& frame, std::vector< std::vector < cv::Rect  > > &rects) try
{
//	dlib::radial_basis_kernel< dlib::matrix < double, 1980, 1 >  > hog;
	//dlib::image_window hogwin(draw_fhog(detector), "Learned fHOG detector");
		
	std::vector< std::vector < cv::Rect  > > dets (frame.size()) ;
	for (int i = 0; i < frame.size(); i++) {
		cv::Mat trz = frame[i].croppedImg;
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::bgr_pixel> temp(trz);
		dlib::assign_image(img, temp);
//		win.set_image(img);
//		win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));
//		std::cout << "Hit enter to process the next image..." << std::endl;
//		std::cin.get();
	//	if(detector(img).empty())
	//		continue;
		//extract_fhog_features(img, hog);
		//	win.set_image(hog(img));
		//detector.detect()
		 //dets[i] = Utils::vecDlibRectangle2VecOpenCV(detector(img),6000);
		 //dlib::image_window win;
	//	 win.clear_overlay();
//		 win.set_image(img);
//		 win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));
		
	}
	//if(dets.size() > 1)
	//	for(auto det : dets)
//		std::cout << det[0] << std::endl;

	rects = dets;
	dets.clear();
}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}
//dlib::image_window win;
void MyFHog::detect(cv::Mat& frame, std::vector< cv::Rect > &rects) try
{
	//dlib::image_window win;
	//dlib::deserialize("D://Codes//school//backupDT//repo//pedestrian//pedDet.svm") >> detector;
	//scanner.set_detection_window_size(80, 80);
//		auto startTime = std::chrono::high_resolution_clock::now();

//	typedef  dlib::matrix < double, 1980, 1   > kernel;
//	//dlib::array <dlib::array2d < double >> hog;
//	dlib::array<dlib::array2d<double>> hog;
		//dlib::impl_fhog::set_hog();

	
//		auto startTime = std::chrono::high_resolution_clock::now();
		dlib::array2d < dlib::bgr_pixel > img;
		dlib::cv_image<dlib::rgb_pixel> temp(frame);
		dlib::assign_image(img, temp);
		//dlib::impl_fhog::impl_extract_fhog_features(img, hog, 8, 1, 1);
//		auto endTime = std::chrono::high_resolution_clock::now();
//		double time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
//		std::cout << "convert took " << static_cast<float>(time) / CLOCKS_PER_SEC << "s." << std::endl;

	
	
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
	//	rects = Utils::vecDlibRectangle2VecOpenCV(detector(img,0.222),4999);
//		endTime = std::chrono::high_resolution_clock::now();
//		time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
//		std::cout << "DETECT WITH CONV " << time / 1000 << std::endl;
		//win.clear_overlay();
		//win.set_image(img);
	//	win.add_overlay(detector(img), dlib::rgb_pixel(0, 255, 0));

}
catch (std::exception& e)
{
	std::cout << e.what() << std::endl;
}