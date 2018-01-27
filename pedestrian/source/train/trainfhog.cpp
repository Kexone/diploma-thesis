#include "trainfhog.h"
#include "../utils/utils.h"
#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

void TrainFHog::pickBestWindowSize( const std::vector<std::vector<dlib::rectangle> >& boxes, unsigned long& width,	unsigned long& height,	const unsigned long target_size )
/*!
ensures
- Finds the average aspect ratio of the elements of boxes and outputs a width
and height such that the aspect ratio is equal to the average and also the
area is equal to target_size.  That is, the following will be approximately true:
- #width*#height == target_size
- #width/#height == the average aspect ratio of the elements of boxes.
!*/
{
	// find the average width and height
	dlib::running_stats<double> avg_width, avg_height;
	for (unsigned long i = 0; i < boxes.size(); ++i)
	{
		for (unsigned long j = 0; j < boxes[i].size(); ++j)
		{
			avg_width.add(boxes[i][j].width());
			avg_height.add(boxes[i][j].height());
		}
	}

	// now adjust the box size so that it is about target_pixels pixels in size
	double size = avg_width.mean()*avg_height.mean();
	double scale = std::sqrt(target_size / size);

	width = (unsigned long)(avg_width.mean()*scale + 0.5);
	height = (unsigned long)(avg_height.mean()*scale + 0.5);
	// make sure the width and height never round to zero.
	if (width == 0)
		width = 1;
	if (height == 0)
		height = 1;
}

bool TrainFHog::containsAnyBoxes(const std::vector<std::vector<dlib::rectangle> >& boxes)
{
	for (unsigned long i = 0; i < boxes.size(); ++i)
	{
		if (boxes[i].size() != 0)
			return true;
	}
	return false;
}

void TrainFHog::throwInvalidBoxErrorMessage( const std::string& dataset_filename,	const std::vector<std::vector<dlib::rectangle> >& removed,	const unsigned long target_size)
{
	dlib::image_dataset_metadata::dataset data;
	load_image_dataset_metadata(data, dataset_filename);

	std::ostringstream sout;
	sout << "Error!  An impossible set of object boxes was given for training. ";
	sout << "All the boxes need to have a similar aspect ratio and also not be ";
	sout << "smaller than about " << target_size << " pixels in area. ";
	sout << "The following images contain invalid boxes:\n";
	std::ostringstream sout2;
	for (unsigned long i = 0; i < removed.size(); ++i)
	{
		if (removed[i].size() != 0)
		{
			const std::string imgname = data.images[i].filename;
			sout2 << "  " << imgname << "\n";
		}
	}
	throw dlib::error("\n" + dlib::wrap_string(sout.str()) + "\n" + sout2.str());
}



TrainFHog::TrainFHog()
{
}

void TrainFHog::train(std::string posSamples, std::string negSamples)
{

	const std::string parser = "dataset/training.xml";
	const std::string samplesPath = "dataset/imgSamples.txt";

	dlib::array<dlib::array2d<unsigned char> > images;
	

	std::fstream sampleFile(samplesPath);
	std::string oSample;
	//while (sampleFile >> oSample) {
		//dlib::cv_image<TrainFHog::pixel_type> cvTmp(sampleFile);
		//dlib::matrix<TrainFHog::pixel_type> test = dlib::mat(cvTmp);
	//	dstList.push_back(test);

	std::vector<std::vector<dlib::rectangle> > objectLocations, ignore;
	ignore = load_image_dataset(images, objectLocations,parser);

	std::cout << "Number of images loaded: " << images.size() << std::endl;
	std::cout << "Number of obj loaded: " << objectLocations[0].size() << std::endl;

	const unsigned int numFolds = images.size();







	std::vector< int > labels;
	//std::vector< dlib::matrix < pixel_type > > samplesLst;
	//std::vector< array2d < matrix < float, 31, 1 > > > featuresLst;


	//std::cout << "LOADING SAMPLES ... ";
	//Utils::fillSamples2List(posSamples, samplesLst, labels, cv::Size(96, 48));
	//Utils::fillSamples2List(negSamples, samplesLst, labels, cv::Size(96,48), true);
	//std::cout << samplesLst.size() << " items." << std::endl;

	//typedef dlib::matrix<float,31,1> sample_type;
//	typedef radial_basis_kernel<sample_type> kernel_type;
	//svm_c_trainer<kernel_type> trainer;
	typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > image_scanner_type;
	const unsigned long targetSize= 96 * 48;
	image_scanner_type scanner;
	unsigned long width, height;
	pickBestWindowSize(objectLocations, width, height, targetSize);
	scanner.set_detection_window_size(width, height);
	dlib::structural_object_detection_trainer<image_scanner_type> trainer(scanner);


	trainer.be_verbose();
	trainer.set_c(0.15325);    // 0.15625
	trainer.set_epsilon(0.001); // 0.001   91.6 %
	trainer.set_num_threads(8);
	
	const unsigned long upsample_amount = 0;
	std::vector<std::vector<dlib::rectangle> > removed;
	removed = remove_unobtainable_rectangles(trainer, images, objectLocations);
	// if we weren't able to get all the boxes to match then throw an error 
	if (containsAnyBoxes(removed))
	{
		unsigned long scale = upsample_amount + 1;
		scale = scale*scale;
		throwInvalidBoxErrorMessage(parser, removed, targetSize / scale);
	}
	std::vector<std::vector<dlib::rectangle>> rects;
	//object_detector<image_scanner_type> detector = trainer.train(images, object_locations, ignore);

	//cout << "Saving trained detector to object_detector.svm" << endl;
	//serialize("object_detector.svm") << detector;

	//cout << "Testing detector on training data..." << endl;
	//cout << "Test detector (precision,recall,AP): " << test_object_detection_function(detector, images, object_locations, ignore) << endl;
	
	
	randomize_samples(images, objectLocations);

	std::cout << numFolds << "-fold cross validation (precision,recall,AP): "
		<< cross_validate_object_detection_trainer(trainer, images, objectLocations, ignore, numFolds) << std::endl;
	
	//typedef scan_fhog_pyramid<pyramid_down<6>> image_scanner_type;
	//image_scanner_type scanner;
	//upsample_image_dataset<pyramid_down<2>>()
	//dlib::array2d<dlib::rgb_pixel> samples;

	//vector_normalizer<sample_type> normalizer;
	//// let the normalizer learn the mean and standard deviation of the samples
	//normalizer.train(samplesLst);
	//// now normalize each sample
	//for (unsigned long i = 0; i < samplesLst.size(); ++i)
	//	samplesLst[i] = normalizer(samplesLst[i]);
	//
	/*for (auto &sample : samplesLst) {
		array2d<matrix<float, 31, 1> > hog;
		extract_fhog_features(sample, hog);
		featuresLst.push_back(hog);
	}*/

}

void TrainFHog::testParams(std::vector<cv::Mat> samplesList, std::vector<int> labels)
{
	typedef dlib::matrix < float, 1980, 1 > sample_type;
	typedef dlib::radial_basis_kernel< sample_type > kernel_type;
	std::vector < sample_type > samples;
	std::vector < float > flLabels(labels.begin(), labels.end());
	dlib::svm_nu_trainer < kernel_type > trainer;

	const double max_nu = 1;

	for(auto &sample :  samplesList)
	{
		dlib::cv_image<float> cvTmp(sample);
		dlib::matrix<float,1980,1> mtxTmp = dlib::mat(cvTmp);
		samples.push_back(mtxTmp);
	}

	std::cout << "All samples : " <<  samples.size() << std::endl;

	std::cout << "doing cross validation" << std::endl;
	for (float gamma = 0.0001; gamma <= 1; gamma *= 5)
	{
		for (float nu = 0.0001; nu < max_nu; nu *= 5)
		{
			// tell the trainer the parameters we want to use
			trainer.set_kernel(kernel_type(gamma));
			trainer.set_nu(nu);
		//	trainer.set_cache_size(samples.size());
			std::cout << "gamma: " << gamma << "    nu: " << nu;
			// Print out the cross validation accuracy for 3-fold cross validation using
			// the current gamma and nu.  cross_validate_trainer() returns a row vector.
			// The first element of the vector is the fraction of +1 training examples
			// correctly classified and the second number is the fraction of -1 training
			// examples correctly classified.
			std::cout << "     cross validation accuracy: " << cross_validate_trainer(trainer, samples, flLabels, 3);
		}
	}
		//typedef dlib::decision_function<kernel_type> dec_funct_type;
		//typedef dlib::normalized_function<dec_funct_type> funct_type;
		//funct_type learned_function;
		//learned_function.normalizer = normalizer;
		//learned_function.function = trainer.train(samples, flLabels);
		
		typedef dlib::probabilistic_decision_function<kernel_type> probabilistic_funct_type;
		typedef dlib::normalized_function<probabilistic_funct_type> pfunct_type;
		trainer.set_kernel(kernel_type(0.001));
		trainer.set_nu(0.001);
		pfunct_type learned_pfunct;
//		learned_pfunct.normalizer = normalizer;
		learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, flLabels, 3);
		typedef dlib::scan_fhog_pyramid<sample_type > image_scanner_type;
		dlib::serialize("data.svm") <<  trainer.train(samples, flLabels);
		//dlib::object_detector<image_scanner_type> detector = 
		//dlib::serialize("data.dat") << learned_pfunct;
		//learned_pfunct.function = train_probabilistic_decision_function(reduced2(trainer, 10), samples, flLabels, 3);
		//dlib::serialize("data.svm") << learned_pfunct;

	//}
}

void bs()
{
//	typedef matrix<double, 2, 1> sample_type;
//	typedef radial_basis_kernel<sample_type> kernel_type;
//
//	std::vector<sample_type> samples;
//	std::vector<double> labels;
//
//	for (int r = -20; r <= 20; ++r)
//	{
//		for (int c = -20; c <= 20; ++c)
//		{
//			sample_type samp;
//			samp(0) = r;
//			samp(1) = c;
//			samples.push_back(samp);
//
//			if (sqrt((double)r*r + c*c) <= 10)
//				labels.push_back(+1);
//			else
//				labels.push_back(-1);
//
//		}
//	}
//
//	vector_normalizer<sample_type> normalizer;
//	normalizer.train(samples);
//	for (unsigned long i = 0; i < samples.size(); ++i)
//		samples[i] = normalizer(samples[i]);
//
//	randomize_samples(samples, labels);
//	svm_c_trainer<kernel_type> trainer;
//
//	
//
//	trainer.set_kernel(kernel_type(0.15625));
//	trainer.set_c(5);
//	typedef decision_function<kernel_type> dec_funct_type;
//	typedef normalized_function<dec_funct_type> funct_type;
//
//	funct_type learned_function;
//	learned_function.normalizer = normalizer;  // save normalization information
//	learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results
//
//																// print out the number of support vectors in the resulting decision function
//	cout << "\nnumber of support vectors in our learned_function is "
//		<< learned_function.function.basis_vectors.size() << endl;
//
//	sample_type sample;
//
//	sample(0) = 3.123;
//	sample(1) = 2;
//	cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;
//
//	sample(0) = 3.123;
//	sample(1) = 9.3545;
//	cout << "This is a +1 class example, the classifier output is " << learned_function(sample) << endl;
//
//	sample(0) = 13.123;
//	sample(1) = 9.3545;
//	cout << "This is a -1 class example, the classifier output is " << learned_function(sample) << endl;
//
//	sample(0) = 13.123;
//	sample(1) = 0;
//	cout << "This is a -1 class example, the classifier output is " << learned_function(sample) << endl;
//
//	typedef probabilistic_decision_function<kernel_type> probabilistic_funct_type;
//	typedef normalized_function<probabilistic_funct_type> pfunct_type;
//
//	pfunct_type learned_pfunct;
//	learned_pfunct.normalizer = normalizer;
//	learned_pfunct.function = train_probabilistic_decision_function(trainer, samples, labels, 3);
//
//	cout << "\nnumber of support vectors in our learned_pfunct is "
//		<< learned_pfunct.function.decision_funct.basis_vectors.size() << endl;
//
//	sample(0) = 3.123;
//	sample(1) = 2;
//	cout << "This +1 class example should have high probability.  Its probability is: "
//		<< learned_pfunct(sample) << endl;
//
//	sample(0) = 3.123;
//	sample(1) = 9.3545;
//	cout << "This +1 class example should have high probability.  Its probability is: "
//		<< learned_pfunct(sample) << endl;
//
//	sample(0) = 13.123;
//	sample(1) = 9.3545;
//	cout << "This -1 class example should have low probability.  Its probability is: "
//		<< learned_pfunct(sample) << endl;
//
//	sample(0) = 13.123;
//	sample(1) = 0;
//	cout << "This -1 class example should have low probability.  Its probability is: "
//		<< learned_pfunct(sample) << endl;
//
//	serialize("saved_function.dat") << learned_pfunct;
//	deserialize("saved_function.dat") >> learned_pfunct;
//
//	cout << "\ncross validation accuracy with only 10 support vectors: "
//		<< cross_validate_trainer(reduced2(trainer, 10), samples, labels, 3);
//	cout << "cross validation accuracy with all the original support vectors: "
//		<< cross_validate_trainer(trainer, samples, labels, 3);
//
//	learned_function.function = reduced2(trainer, 10).train(samples, labels);
//	learned_pfunct.function = train_probabilistic_decision_function(reduced2(trainer, 10), samples, labels, 3);
}
