#include "extractorROI.h"
#include "utils.h"


void ExtractorROI::extractROI(std::string videoStreamPath)
{
	std::string color = "\033[7;49;91m";
	std::string greenColor = "\033[0;49;92m";
	std::string yellowColor = "\033[0;49;93m";
	std::string grayColor = "\033[0;49;90m";
	std::string redBackground = "\033[0;41m";

	unsigned first = videoStreamPath.find("/");
	unsigned last = videoStreamPath.find(".");
	path = videoStreamPath.substr(first+1, last - first-1);
	
	Utils::makeDir(path+"\\ROI");
	
	vs = new VideoStream(videoStreamPath);
	vs->openCamera();

	int countFrame = 0;
	int totalFrames = VideoStream::totalFrames;
	rects2Save = std::vector < std::vector < cv::Rect > >(totalFrames);
	ROIs = std::vector< cv::Mat >(N_RECT);
	rects.clear();

	for (int i = 0; i < N_RECT; i++)
		rects.push_back(cv::Rect());

	std::cout << std::endl;
	std::cout << yellowColor << "'s' save\n";
	std::cout << yellowColor << "'n' next frame" << std::endl;
	std::cout << yellowColor << "'r' to reset current rect position" << std::endl;
	std::cout << yellowColor << "'x' immediately write to file (not recommend)" << std::endl << std::endl;
	std::cout << yellowColor << "'j' move actual rect to left" << std::endl;
	std::cout << yellowColor << "'l' move actual rect to right" << std::endl;
	std::cout << yellowColor << "'i' move actual rect to up" << std::endl;
	std::cout << yellowColor << "'k' move actual rect to down" << std::endl;
	std::cout << yellowColor << "'J' decrease the left" << std::endl;
	std::cout << yellowColor << "'L' increase the left" << std::endl;
	std::cout << yellowColor << "'I' decrease the bottom" << std::endl;
	std::cout << yellowColor << "'K' increase the bottom" << std::endl;

	std::cout << yellowColor << "'0-" << N_RECT - 1 << "' changes active rect" << std::endl << std::endl;
	std::cout << grayColor << "\nVideostream initialized.\n\n" << std::endl;

	while (true) {
		cv::Mat frame = vs->getFrame();
		if (frame.empty()) {
			
			write2File();
			delete vs;
			break;
		}
		fullFrame = frame.clone();
		img = frame.clone();
		std::cout  << greenColor << countFrame << "/" << totalFrames << " FRAME " << std::endl;
		process(countFrame);
		countFrame++;
	}
	std::cout << "DONE!" << std::endl;
}

void ExtractorROI::write2File()
{
	std::ofstream fs;
	fs.open(path+".txt", std::ios::app);
	int ind = 0;
	fs << rects2Save.size() << std::endl;

	for (auto rects : rects2Save)	{
		for (int i = 0; i < rects.size(); i++)		{
			if (rects[i].area() == 0) continue;
			fs << ind << " "<< rects[i].tl().x << " " << rects[i].tl().y << " " << rects[i].br().x  << " " << rects[i].br().y << std::endl;
		}
		ind++;
	}
	fs.close();
}

void ExtractorROI::onMouse(int event, int x, int y, int, void* userdata)
{
	ExtractorROI* extract = reinterpret_cast<ExtractorROI*>(userdata);
	extract->onMouse(event, x, y);
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
	namedWindow(WIN_NAME, cv::WINDOW_AUTOSIZE);
	cv::setMouseCallback(WIN_NAME, onMouse, this);
	indRect = 0;
	point1 = cv::Point(0, 0);
	point2 = cv::Point(0, 0);
	
	cv::imshow(WIN_NAME, fullFrame);
	std::cout << "\033[0;49;90m" << "\tActive " << indRect << " rect" << std::endl;
	bool nextOp = true;
	while (nextOp) {
		char c = cv::waitKey();
		switch (c) {
		case 'i':	rects[indRect].y--;
			break;
		case 'j':	rects[indRect].x--;
			break;
		case 'k':	rects[indRect].y++;
			break;
		case 'l':	rects[indRect].x++;
			break;
		case 'I':	rects[indRect].height--;
			break;
		case 'J':	rects[indRect].width--;
			break;
		case 'K':	rects[indRect].height++;
			break;
		case 'L':	rects[indRect].width++;
			break;
		case 'n':	rects2Save[cFrame] = { cv::Rect(0, 0, 0, 0) };
					nextOp = false;
			break;
		case 's':
			std::sprintf(imgName, "%d_full.jpg", cFrame);
			cv::imwrite(path + "\\" + imgName, img);
			for (int i = 0; i < ROIs.size(); i++) {
				if (ROIs[i].rows == 0) continue;
				std::sprintf(imgName, "%d_roi_%d.jpg", cFrame, i);
				cv::imwrite(path + "\\ROI\\" + imgName, ROIs[i]);
				cv::destroyWindow("cropped_" + i);
				nextOp = false;
			}
			rects2Save[cFrame] = rects;
			std::cout << "\033[0;49;90m" << "\tSaved" << std::endl;
			break;
		case 'r':  
			rects[indRect].x = 0; rects[indRect].y = 0; rects[indRect].width = 0; rects[indRect].height = 0;
			img = fullFrame.clone();
			break;
		case 'x':	
			write2File();
			nextOp = false;
			break;
		}
		int numb =  static_cast<int>(c - '0');
		if(numb < N_RECT && numb >=0)
		{
			indRect = numb;
			std::cout << "\033[0;49;97m" << "\tActive " << indRect << " rect" << std::endl;
		//	std::cout << rects[indRect].tl().x << " " << rects[indRect].tl().y << " " << rects[indRect].br().x << " " << rects[indRect].br().y << std::endl;
		}
		/*else {
			std::cout << "Too much large number or pressed bad key. " << std::endl;
		}*/
		showImage();
		drawRects();
	}
}

void ExtractorROI::showImage() {
	if(clicked)
		img = fullFrame.clone();
	cv::rectangle(img, rects[indRect], rectColors[indRect], 1, 8, 0);
	cv::imshow(WIN_NAME, img);
}

void ExtractorROI::drawRects()
{
	img = fullFrame.clone();
	checkBoundary();

	for (int i = 0; i < rects.size(); i++)	{
		std::stringstream ss;
		ss << "cropped_" << i;
		if( i == indRect)
			cv::destroyWindow(ss.str().c_str());

		if (rects[i].width > 0 && rects[i].height > 0)	{
			ROIs[i] = fullFrame(rects[i]);
			cv::imshow(ss.str().c_str(), ROIs[i]);
			cv::putText(img, labels[i], cv::Point(rects[i].tl().x, rects[i].tl().y-5), cv::FONT_HERSHEY_SIMPLEX,0.5, rectColors[i], 1, cv::LINE_8);
			cv::rectangle(img, rects[i], rectColors[i], 1, 8, 0);
		}
	}
}

void ExtractorROI::checkBoundary() {
	if (rects[indRect].width>fullFrame.cols - rects[indRect].x)		rects[indRect].width = fullFrame.cols - rects[indRect].x;
	if (rects[indRect].height>fullFrame.rows - rects[indRect].y)	rects[indRect].height = fullFrame.rows - rects[indRect].y;
	if (rects[indRect].x<0)	rects[indRect].x = 0;
	if (rects[indRect].y<0)	rects[indRect].height = 0;
}