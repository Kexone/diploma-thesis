#include "dlibSvmTest.h"

cv::Vec4f DlibSvmTest::process()
{
	typedef dlib::matrix < float, 1980, 1 > sample_type;
	typedef dlib::radial_basis_kernel< sample_type > kernel_type;
	std::vector < sample_type > samples;
	
	dlib::svm_nu_trainer < kernel_type > trainer;

	const double max_nu = 1;

	for (auto &sample : samplesList)
	{
		dlib::cv_image<float> cvTmp(sample);
		dlib::matrix<float, 1980, 1> mtxTmp = dlib::mat(cvTmp);
		samples.push_back(mtxTmp);
	}

	cv::Vec4f vec(0.f,0.f,0.f,0.f);
	std::cout << "doing cross validation" << std::endl;
	for (float gamma = 0.0001; gamma <= 1; gamma *= 5)
	{
		for (float nu = 0.0001; nu < max_nu; nu *= 5)
		{
			cv::Vec2f testVec;

			trainer.set_kernel(kernel_type(gamma));
			trainer.set_nu(nu);
			trainer.set_cache_size(samples.size());
			
			std::cout << "gamma: " << gamma << "    nu: " << nu;
			dlib::matrix<float, 1980, 1> test = cross_validate_trainer(trainer, samples, fLabels, 3);
			
			testVec[0] = test(0);
			testVec[1] = test(1);

			if(testVec[0] > vec[0] && testVec[1] > vec[1])
			{
				vec[0] = std::move(testVec[0]);
				vec[1] = std::move(testVec[1]);
				vec[2] = gamma;
				vec[3] = nu;
			}
			std::cout << "     cross validation accuracy: " << vec[0] << "\t" << vec[1] << std::endl;
		}
	}
	writeResult2File(vec);
	return vec;
}

void DlibSvmTest::writeResult2File(cv::Vec4f resultVec)
{
	std::ofstream file;
	file.open("data/testing_dlib_result.txt", std::ios::app);

	file << "RESULT OF DLIB SVM TEST\n";
	file << "THE BEST PARAMETERS:\n";
	file << "GAMMA " << resultVec[2];
	file << "\nNU " << resultVec[3];
	file << "\nPositive results: " << resultVec[1];
	file << ",\t negative results: " << resultVec[2];
	file.close();
}