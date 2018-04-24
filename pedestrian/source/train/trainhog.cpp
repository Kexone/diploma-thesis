#include "trainhog.h"



TrainHog::TrainHog()
{
	this->maxIterations = Settings::maxIterations;
	this->termCriteria = Settings::termCriteria;
	this->kernel = Settings::kernel;
	this->type = Settings::type;
	this->epsilon = Settings::epsilon;
	this->coef0 = Settings::coef0;
	this->degree = Settings::degree;
	this->gamma = Settings::gamma;
	this->nu = Settings::paramNu;
	this->p = Settings::paramP;
	this->c = Settings::paramC;
	this->classifierName = Settings::classifierName2Train + ".yml";
	this->pedestrianSize = Settings::pedSize;
	this->blockSize = Settings::blockSize;
	this->cellSize = Settings::cellSize;
	this->strideSize = Settings::strideSize;
}

TrainHog::TrainHog(int maxIterations, int termCriteria, int kernel, int type, double epsilon, double coef0,
	int degree, double gamma, double nu, double p, double c, std::string classifierName)
{
	this->maxIterations = maxIterations;
	this->termCriteria = termCriteria;
	this->kernel = kernel;
	this->type = type;
	this->epsilon = epsilon;
	this->coef0 = coef0;
	this->degree = degree;
	this->gamma = gamma;
	this->nu = nu;
	this->p = p;
	this->c = c;
	this->classifierName = classifierName;
	this->pedestrianSize = Settings::pedSize;
	this->blockSize = Settings::blockSize;
	this->cellSize = Settings::cellSize;
	this->strideSize = Settings::strideSize;
}

void TrainHog::trainFromMat(std::string matPath, std::string labelsPath)
{
	cv::Mat trainMat;
	cv::FileStorage fs(matPath, cv::FileStorage::READ);
	fs["samples"] >> trainMat;

	std::ifstream is(labelsPath);
	std::istream_iterator < int > start(is), end;
	std::vector< int > labels = std::vector < int > (start, end);

	trainSvm(trainMat, labels);
}

void TrainHog::trainFromMat(cv::Mat trainMat, std::vector<int> labels)
{
	trainSvm(trainMat, labels);
}

void TrainHog::train(bool saveData, bool trainTwice)
{
	std::vector< cv::Mat > posSamplesLst;
	std::vector< cv::Mat > negSamplesLst;
    std::vector< cv::Mat > gradientLst;
	std::vector< int > labels;
    cv::Mat trainMat;

	Utils::fillSamples2List(Settings::samplesPos, posSamplesLst, pedestrianSize);

	labels.assign(posSamplesLst.size(), +1);
	const unsigned int old = static_cast<unsigned int>(labels.size());

	Utils::fillSamples2List(Settings::samplesNeg, negSamplesLst, pedestrianSize);

	labels.insert(labels.end(), negSamplesLst.size(), 0);
	CV_Assert(old < labels.size());

	std::cout << "Positive samples: " << posSamplesLst.size() << std::endl;
	std::cout << "Negative samples: " << negSamplesLst.size() << std::endl;

    extractFeatures(posSamplesLst, gradientLst);
    extractFeatures(negSamplesLst, gradientLst);
    convertSamples2Mat(gradientLst, trainMat);

	if (saveData) saveLabeledMat(trainMat, labels);

    trainSvm(trainMat, labels);

	if (trainTwice)
	{
		cv::Size pos_image_size = posSamplesLst[0].size();
		std::cout << "Testing trained detector on negative images.this may take a few minutes...";
		cv::HOGDescriptor my_hog;
		my_hog.winSize = pos_image_size;
		std::vector<cv::Rect > locations;

		// Set the trained svm to my_hog
		std::vector< float > hog_detector;
		
		getSvmDetector(svm, hog_detector);
		my_hog.setSVMDetector(hog_detector);

		std::vector<cv::Rect> detections;
		std::vector<double> foundWeights;

		for (size_t i = 0; i < negSamplesLst.size(); i++)
		{
			my_hog.detectMultiScale(negSamplesLst[i], detections);
			for (size_t j = 0; j < detections.size(); j++)
			{
				cv::Mat detection = negSamplesLst[i](detections[j]).clone();
				resize(detection, detection, pos_image_size);
				negSamplesLst.push_back(detection);
			}
		}
		std::cout << "...[done]" << std::endl;

		labels.clear();
		labels.assign(posSamplesLst.size(), +1);
		labels.insert(labels.end(), negSamplesLst.size(), 0);

		gradientLst.clear();
		std::cout << "Histogram of Gradients are being calculated for positive images...";
		extractFeatures(posSamplesLst, gradientLst);
		std::cout << "...[done]" << std::endl;

		std::cout << "Histogram of Gradients are being calculated for negative images...";
		extractFeatures(negSamplesLst, gradientLst);
		std::cout << "...[done]" << std::endl;

		std::cout << "Training SVM again...";
		convertSamples2Mat(gradientLst, trainMat);
		trainSvm(trainMat, labels);
		std::cout << "...[done]" << std::endl;
	}
	//std::vector< float > hog_detector;
	//cv::HOGDescriptor hog;
	//hog.winSize = pedestrianSize;
	//getSvmDetector(svm, hog_detector);
	//hog.setSVMDetector(hog_detector);
	//hog.save("my_detector.yml");
}

void TrainHog::calcMatForTraining(cv::Mat& trainMat, std::vector<int> &labels, bool isDlib)
{
	std::vector< cv::Mat > posSamplesLst;
	std::vector< cv::Mat > negSamplesLst;
	std::vector< cv::Mat > gradientLst;

	Utils::fillSamples2List(Settings::samplesPos, posSamplesLst, pedestrianSize);

	labels.assign(posSamplesLst.size(), +1);
	const unsigned int old = static_cast<unsigned int>(labels.size());

	Utils::fillSamples2List(Settings::samplesNeg, negSamplesLst, pedestrianSize);

	if(isDlib)
		labels.insert(labels.end(), negSamplesLst.size(), -1);
	else
		labels.insert(labels.end(), negSamplesLst.size(), 0);
	CV_Assert(old < labels.size());

	std::cout << "Positive samples: " << posSamplesLst.size() << std::endl;
	std::cout << "Negative samples: " << negSamplesLst.size() << std::endl;

	extractFeatures(posSamplesLst, gradientLst);
	extractFeatures(negSamplesLst, gradientLst);
	convertSamples2Mat(gradientLst, trainMat);
}

// From https://github.com/opencv/opencv/blob/ddf82d0b154873510802ef75c53e628cd7b2cb13/samples/cpp/train_HOG.cpp#L134
cv::Mat get_hogdescriptor_visu(const cv::Mat& color_origImg, std::vector<float>& descriptorValues, const cv::Size & size)
{
	const int DIMX = size.width;
	const int DIMY = size.height;
	float zoomFac = 3;
	cv::Mat visu;
	cv::resize(color_origImg, visu, cv::Size((int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac)));

	int cellSize = 8;
	int gradientBinSize = 9;
	float radRangeForOneBin = (float)(CV_PI / (float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

																	   // prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = DIMX / cellSize;
	int cells_in_y_dir = DIMY / cellSize;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin<gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx<blocks_in_x_dir; blockx++)
	{
		for (int blocky = 0; blocky<blocks_in_y_dir; blocky++)
		{
			// 4 cells per block ...
			for (int cellNr = 0; cellNr<4; cellNr++)
			{
				// compute corresponding cell nr
				cellx = blockx;
				celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3)
				{
					cellx++;
					celly++;
				}

				for (int bin = 0; bin<gradientBinSize; bin++)
				{
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)


				  // note: overlapping blocks lead to multiple updates of this sum!
				  // we therefore keep track how often a cell was updated,
				  // to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)


	  // compute average gradient strengths
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (celly = 0; celly<cells_in_y_dir; celly++)
	{
		for (cellx = 0; cellx<cells_in_x_dir; cellx++)
		{
			int drawX = cellx * cellSize;
			int drawY = celly * cellSize;

			int mx = drawX + cellSize / 2;
			int my = drawY + cellSize / 2;

			rectangle(visu, cv::Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), cv::Point((int)((drawX + cellSize)*zoomFac), (int)((drawY + cellSize)*zoomFac)), cv::Scalar(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin<gradientBinSize; bin++)
			{
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = (float)(cellSize / 2.f);
				float scale = 2.5; // just a visualization scale, to see the lines better

								   // compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visualization
				line(visu, cv::Point((int)(x1*zoomFac), (int)(y1*zoomFac)), cv::Point((int)(x2*zoomFac), (int)(y2*zoomFac)), cv::Scalar(0, 255, 0), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	  // don't forget to free memory allocated by helper data structures!
	for (int y = 0; y<cells_in_y_dir; y++)
	{
		for (int x = 0; x<cells_in_x_dir; x++)
		{
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visu;

} // get_hogdescriptor_visu

void TrainHog::extractFeatures(const std::vector< cv::Mat > &samplesLst, std::vector< cv::Mat > &gradientLst) const
{
	cv::HOGDescriptor my_hog;
	my_hog.winSize = pedestrianSize;
	my_hog.gammaCorrection = true;
    cv::HOGDescriptor hog(
					pedestrianSize, //winSize
                    cv::Size(blockSize,blockSize), //blocksize
                    cv::Size(strideSize,strideSize), //blockStride,
                    cv::Size(cellSize,cellSize), //cellSize,
                    9, //nbins,
                    0, //derivAper,
                    -1, //winSigma,
                    0, //histogramNormType,
                    0.2, //L2HysThresh,
                    true //gammal corRection,
                                    //nlevels=64
                    );
	cv::Rect r = cv::Rect(0, 0, pedestrianSize.width, pedestrianSize.height);
	r.x += (samplesLst[0].cols - r.width) / 2;
	r.y += (samplesLst[0].rows - r.height) / 2;
    cv::Mat gr;
    std::vector< cv::Point > location;
    std::vector< float > descriptors;
    for(auto &mat : samplesLst) {
		
		cv::cvtColor(mat(r), gr, cv::COLOR_BGR2GRAY);
        //cv::cvtColor(mat, gr,cv::COLOR_BGR2GRAY);
		//cv::equalizeHist(gr, gr);
	//	gr.convertTo(gr, CV_8U);

		my_hog.compute(gr, descriptors,cv::Size(8,8),cv::Size(0, 0), location);
	   // cv::imshow("gradient", get_hogdescriptor_visu(mat, descriptors, gr.size()));
	  //  cv::waitKey(10);
		gradientLst.push_back( cv::Mat(descriptors).clone() );
    }
}

void TrainHog::trainSvm(cv::Mat &trainMat, const std::vector<int> &labels)
{
  //  std::cout << "START training ..." << std::endl;
	clock_t timer = clock();
	svm = cv::ml::SVM::create();
	//cv::Ptr<cv::ml::Boost> svm = cv::ml::Boost::create();

	svm->setCoef0(this->coef0);
	svm->setDegree(this->degree);
	svm->setTermCriteria(cv::TermCriteria(this->termCriteria, this->maxIterations, this->epsilon));
	svm->setGamma(this->gamma);
	svm->setKernel(this->kernel);
	svm->setNu(this->nu);
	svm->setP(this->p);
	svm->setC(this->c);
	svm->setType(this->type);
	svm->train(trainMat, cv::ml::ROW_SAMPLE, cv::Mat(labels));
	svm->save(this->classifierName);
	timer = clock() - timer;
 //   std::cout << "training DONE ..."<< static_cast<float>(timer) / (CLOCKS_PER_SEC*60) << " min" <<  std::endl;

}

void TrainHog::convertSamples2Mat(const std::vector<cv::Mat> &trainSamples, cv::Mat &trainData)
{
	//--Convert data
	const int rows = (int)trainSamples.size();
	const int cols = (int)std::max(trainSamples[0].cols, trainSamples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < trainSamples.size(); ++i)
	{
		CV_Assert(trainSamples[i].cols == 1 || trainSamples[i].rows == 1);

		if (trainSamples[i].cols == 1)
		{
			transpose(trainSamples[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (trainSamples[i].rows == 1)
		{
			trainSamples[i].copyTo(trainData.row((int)i));
		}
	}
}

void TrainHog::saveLabeledMat(cv::Mat data, std::vector< int > labels)
{
	cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
	fs << "samples"<< data;
	fs.release();
	std::ofstream output_file("./labels.txt");
	std::ostream_iterator<int> output_iterator(output_file, "\n");
	std::copy(labels.begin(), labels.end(), output_iterator);
}

void TrainHog::getSvmDetector(const cv::Ptr<cv::ml::SVM>& svm, std::vector<float>& hog_detector)
{
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

void TrainHog::printSettings()
{
	std::cout << std::setw(10) << std::setfill('_') << std::endl;
	std::cout << "SVM SETTING" << std::endl;
	std::cout << "MAX ITER: " << this->maxIterations << std::endl;
	std::cout << "TERM CRIT: " << this->termCriteria << std::endl;
	std::cout << "KERNEL: " << this->kernel << std::endl;
	std::cout << "TYPE SVM: " << this->type << std::endl;
	std::cout << "EPSILON: " << this->epsilon << std::endl;
	std::cout << "COEF0:" << this->coef0 << std::endl;
	std::cout << "DEGREE: " << this->degree << std::endl;
	std::cout << "GAMMA: " << this->gamma << std::endl;
	std::cout << "NU: " << this->nu << std::endl;
	std::cout << "P: " << this->p << std::endl;
	std::cout << "C: " << this->c << std::endl;
	std::cout << std::setw(10) << std::setfill('_') << std::endl;
}
