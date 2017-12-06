#include "extractorROI.h"



void ExtractorROI::extractROI(std::string videoStreamPath)
{
	hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, 14);
	std::cout << "\n's' save" << std::endl;
	std::cout << "'n' next frame"  << std::endl;
	std::cout << "'r' to reset" << std::endl;
	std::cout << "'x' immediately write to file (not recommend)" << std::endl;
	std::cout << "'0-"<<rectCount-1<<"' changes active rect" << std::endl << std::endl;


	SetConsoleTextAttribute(hConsole, 8);
	unsigned first = videoStreamPath.find("/");
	unsigned last = videoStreamPath.find(".");
	path = videoStreamPath.substr(first+1, last - first-1);
	if (!std::experimental::filesystem::exists(path))	{
		std::experimental::filesystem::create_directory(path);
		std::experimental::filesystem::create_directory(path + "\\ROI");
	}
	vs = new VideoStream(videoStreamPath);
	vs->openCamera();

	std::cout << "Videostream initialized.\n\n" << std::endl;

	int countFrame = 0;
	int totalFrames = VideoStream::totalFrames;
	rects2Save = std::vector < std::vector < cv::Rect > >(totalFrames);
	ROIs = std::vector< cv::Mat >(rectCount);
	while (true) {
		cv::Mat frame = vs->getFrame();
		if (frame.empty()) {
			
			write2File();
			delete vs;
			break;
		}
		fullFrame = frame.clone();
		img = frame.clone();
		SetConsoleTextAttribute(hConsole, 10);
		std::cout  << countFrame << "/" << totalFrames << " FRAME " << std::endl;
		SetConsoleTextAttribute(hConsole, 8);
		process(countFrame);
		countFrame++;
	}
	std::cout << "DONE!" << std::endl;
}

void ExtractorROI::write2File()
{
	std::ofstream fs;
	fs.open(nameFile, std::ios::app);
	fs << "************************************" << std::endl;
	fs << "Total frames: " <<  VideoStream::totalFrames << "FPS: " << VideoStream::fps << std::endl << std::endl;
	int ind = 0;
	for (auto rects : rects2Save)
	{
		fs << ++ind << " FRAME" << std::endl << "\t";
		for (int i = 0; i < rects.size(); i++)
		{
			fs << i << " "<< rects[i].tl() << " " << rects[i].br() << " ";
		}
		fs << std::endl;
	}
	fs << "END OF STREAM" << std::endl;
	fs.close();
}


void ExtractorROI::onMouse(int event, int x, int y) {
	switch (event) {
		case  CV_EVENT_LBUTTONDOWN:
			clicked = true;

			point1.x = x;
			point1.y = y;
			point2.x = x;
			point2.y = y;
			break;

		case  CV_EVENT_LBUTTONUP:
			point2.x = x;
			point2.y = y;
			clicked = false;
			drawRects();
			break;

		case  CV_EVENT_MOUSEMOVE:
			if (clicked) {
				point2.x = x;
				point2.y = y;
			}
			break;

		default:   break;
	}

	if (clicked) {
		if (point1.x>point2.x) {
			rects[indRect].x = point2.x;
			rects[indRect].width = point1.x - point2.x;
		}
		else {
			rects[indRect].x = point1.x;
			rects[indRect].width = point2.x - point1.x;
		}

		if (point1.y>point2.y) {
			rects[indRect].y = point2.y;
			rects[indRect].height = point1.y - point2.y;
		}
		else {
			rects[indRect].y = point1.y;
			rects[indRect].height = point2.y - point1.y;
		}
	}
	showImage();
}

void ExtractorROI::process(int cFrame)
{
	indRect = 0;
	point1 = cv::Point(0, 0);
	point2 = cv::Point(0, 0);
	rects.clear();

	for (int i = 0; i < rectCount; i++)
		rects.push_back(cv::Rect(0, 0, 0, 0));

	namedWindow(winName, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(winName, onMouse, this);
	cv::imshow(winName, fullFrame);
	std::cout << "\tActive " << indRect << " rect" << std::endl;
	while (true) {
		char c = cv::waitKey();
		if (c == 's') {
			std::sprintf(imgName, "%d_full.jpg", cFrame);
			cv::imwrite(path + "\\" + imgName, img);
			for (int i = 0; i < ROIs.size(); i++) {
				if (ROIs[i].rows == 0) continue;
				std::sprintf(imgName, "%d_roi_%d.jpg", cFrame,i);
				cv::imwrite(path + "\\ROI\\" + imgName, ROIs[i]);
				cv::destroyWindow("cropped_" + i);
			}
			rects2Save[cFrame] = rects;
			std::cout << "\tSaved" << std::endl;
			break;
		}
		if (c == 'n') {
			rects2Save[cFrame] = { cv::Rect(0, 0, 0, 0) };
			break;
		}
		if (c == 'x') {
			write2File();
			break;
		}
		if (c == 'r') { rects[indRect].x = 0; rects[indRect].y = 0; rects[indRect].width = 0; rects[indRect].height = 0; }

		int numb =  static_cast<int>(c - '0');
		if(numb < rectCount && numb >=0)
		{
			indRect = numb;
			SetConsoleTextAttribute(hConsole, 15);
			std::cout << "\tnnActive " << indRect << " rect" << std::endl;
			SetConsoleTextAttribute(hConsole, 8);
		}
		else {
			SetConsoleTextAttribute(hConsole, 12);
			std::cout << "Too much large number or pressed bad key. " << std::endl;
			SetConsoleTextAttribute(hConsole, 8);
		}
		showImage();
		drawRects();
	}
}

void ExtractorROI::showImage() {
	if(clicked)
		img = fullFrame.clone();
	cv::rectangle(img, rects[indRect], cv::Scalar(0, 255, 0), 1, 8, 0);
	cv::imshow(winName, img);
}

void ExtractorROI::drawRects()
{
	img = fullFrame.clone();
	checkBoundary();

	for (int i = 0; i < rects.size(); i++)	{
		if (rects[i].width > 0 && rects[i].height > 0)	{
			ROIs[i] = fullFrame(rects[i]);
			cv::imshow("cropped_"+i, ROIs[i]);
			cv::rectangle(img, rects[i], cv::Scalar(0, 255, 0), 1, 8, 0);
		}
		else	{
			cv::destroyWindow("cropped_" + i);
		}
	}
}

void ExtractorROI::onMouse(int event, int x, int y, int, void* userdata)
{
	ExtractorROI* extract = reinterpret_cast<ExtractorROI*>(userdata);
	extract->onMouse(event, x, y);
}

void ExtractorROI::checkBoundary() {
	if (rects[indRect].width>fullFrame.cols - rects[indRect].x)		rects[indRect].width = fullFrame.cols - rects[indRect].x;
	if (rects[indRect].height>fullFrame.rows - rects[indRect].y)	rects[indRect].height = fullFrame.rows - rects[indRect].y;
	if (rects[indRect].x<0)	rects[indRect].x = 0;
	if (rects[indRect].y<0)	rects[indRect].height = 0;
}