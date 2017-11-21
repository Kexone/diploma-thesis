#include "hog.h"

Hog::Hog()
{
	hog.gammaCorrection = true;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	std::cout << "Initialized default people detector" << std::endl;
}

Hog::Hog(std::string svmPath)
{
	svm = cv::Algorithm::load<cv::ml::SVM>(svmPath);
	std::vector< float > hogDetector;
	getSvmDetector(svm, hogDetector);
	hog.svmDetector = hogDetector;
	hog.gammaCorrection = true;
	std::cout << "Initialized custom SVM" << std::endl;
}

std::vector<std::vector<cv::Rect>> Hog::detect(std::vector<CroppedImage>& frames) {

	//std::cout << "PIC size: " << frames.size() << std::endl;
	std::vector<std::vector<cv::Rect>> found_filtered(frames.size());

       // fflush(stdout);
        for (uint x = 0; x < frames.size(); x++) {
            std::vector<cv::Rect> rRect;
            std::vector<cv::Rect> found;
			cv::Mat test = frames[x].croppedImg;

            assert(!test.empty());
         //   test.convertTo(test,CV_8UC1);
			cv::cvtColor(test, test, CV_BGR2GRAY);
			cv::equalizeHist(test, test);
			cv::imshow("test", test);
            hog.detectMultiScale(
            						test,					// testing img
            						found,					// foundLocation <rect>
            						1,						// hitThreshold = 0 // 1
            						cv::Size(8, 8),			// winStride size(8, 8)
            						cv::Size(0,0),			// padding size(0, 0)
            						1.05,					// scale = 1,05
            						2,						// finalThreshold = 2 // 0
									false					// use meanshift grouping = false
            				    );

            if (found.empty()) {
                continue;
            }
            size_t i, j;

            for (i = 0; i< found.size(); i++)
            {
                //cv::Rect r = found[i];

                //for (j = 0; j<found.size(); j++)
                //    if (j != i && (r & found[j]) == r)
              //          break;
            //    if (j == found.size())
                    found_filtered[x].push_back(found[i]);
					//  std::cout << "TL" << found[i].tl().x << found[i].tl().y << " BR" << found[i].br().x << found[i].br().y;
          //          cv::rectangle(test, found[i].tl(), found[i].br(),cv::Scalar(0,0,255),4,8,0);
            }
         //   cv::imshow("test", test);
            //found.clear();
        }
        return found_filtered;
}

void Hog::detect(std::vector<cv::Mat> &testLst, int &nTrue, int &nFalse, bool pedestrian)
{
	std::vector< cv::Point > location;
	std::vector< float > descriptors;
	hog.winSize = cv::Size(48, 96);
	for(auto &mat : testLst)
	{
		hog.compute(mat, descriptors, cv::Size(8, 8), cv::Size(0, 0), location);
		int predicted = svm->predict(descriptors);
		if(pedestrian)
		{
			if (predicted == 1) nTrue++;
			else nFalse++;
		}
		else
		{
			if (predicted == 0) nTrue++;
			else nFalse++;
		}
	}
}

void Hog::getSvmDetector( const cv::Ptr< cv::ml::SVM > &svm, std::vector< float > &hog_detector )
{
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    //CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    //CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
      //         (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    //CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
}
