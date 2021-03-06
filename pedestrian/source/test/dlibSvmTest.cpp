﻿#include "dlibSvmTest.h"

DlibSvmTest::DlibSvmTest(cv::Mat trainMat, std::vector<double> labels)
{
	for (int y = 0; y < trainMat.rows; y++)
	{
		sample_type samp;

		for (int x = 0; x <  trainMat.cols; x++)
		{
			double val = 0.0;
			val = trainMat.at<float>(y, x);
			samp(x) = val;

		}
		_samples.push_back(samp);
		_labels.push_back(labels[y]);
	}
}

void DlibSvmTest::process(int type)
{
	double vec[] = { 0,0,0,0 };
	try {
		if (type == 1)
			testNusvm(vec);
		else
			testCsvm(vec);
		writeResult2File(vec, type);
	}
	catch(std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

void DlibSvmTest::writeResult2File(double *vec, int type)
{
	std::ofstream file;
	file.open("data/testing_dlib_result.txt", std::ios::app);

	file << "RESULT OF DLIB SVM TEST\n";
	file << "THE BEST PARAMETERS:\n";
	file << "GAMMA " << vec[0];
	if (type == 1) {
		file << "\nNU " << vec[1];
		file << "\nPositive results: " << vec[2];
		file << ",\t negative results: " << vec[3];
	}	else	{
		file << "\nC1 " << vec[1];
		file << "\nC2 " << vec[2];
	}
	file.close();
}

void DlibSvmTest::testCsvm(double *vec)
{
	auto cross_validation_score = [&](const double gamma, const double c1, const double c2)
	{

		dlib::svm_c_trainer<kernel_type> trainer;
		
		trainer.set_kernel(kernel_type(gamma));
		trainer.set_c_class1(c1);
		trainer.set_c_class2(c2);

		dlib::matrix< double > result = dlib::cross_validate_trainer(trainer, _samples, _labels, 10);
		std::cout << "gamma: " << std::setw(11) << gamma << "  c1: " << std::setw(11) << c1 << "  c2: " << std::setw(11) << c2 << "  cross validation accuracy: " << result;     
		return 2 * dlib::prod(result) / dlib::sum(result);
	};

	auto result = find_max_global(cross_validation_score,
	{ 1e-5, 1e-5, 1e-5 },  // lower bound constraints on gamma, c1, and c2, respectively
	{ 100,  1e6,  1e6 },   // upper bound constraints on gamma, c1, and c2, respectively
	                              dlib::max_function_calls(50));

	vec[0] = result.x(0); // best gamma
	vec[1] = result.x(1); // best c1
	vec[2] = result.x(2); // bect c2
	vec[3] = result.x; // 

	std::cout << " best cross-validation score: " << result.y << std::endl;
	std::cout << " best gamma: " << vec[0] << "   best c1: " << vec[1] << "    best c2: " << vec[2] << std::endl;
}


void DlibSvmTest::testNusvm(double*vec)
{
	dlib::svm_nu_trainer < kernel_type > trainer;

	const double max_nu = 1;
	std::cout << "doing cross validation" << std::endl;
	for (double gamma = 0.0001; gamma <= 1; gamma *= 5)
	{
		for (double nu = 0.0001; nu < max_nu; nu *= 5)
		{
			trainer.set_kernel(kernel_type(gamma));
			trainer.set_nu(nu);
			trainer.set_cache_size(_samples.size());

			std::cout << "gamma: " << gamma << "    nu: " << nu << " " << cross_validate_trainer(trainer, _samples, _labels, 3);
			auto test = cross_validate_trainer(trainer, _samples, _labels, 3);
			cv::Vec2f testVec(0.f, 0.f);

			testVec[0] = test(0);
			testVec[1] = test(1);

			if (testVec[0] > vec[0] && testVec[1] > vec[1])
			{
				vec[2] = std::move(testVec[0]); //accuracy on pos samples
				vec[3] = std::move(testVec[1]); // accuracy on neg samples
				vec[0] = gamma; //gamma
				vec[1] = nu; //nu
			} 
		}
	}
}