#include "trainfhog.h"
#include "../utils/utils.h"
#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>

void TrainFHog::pickBestWindowSize( const std::vector<std::vector<dlib::rectangle> >& boxes, unsigned long& width,	unsigned long& height,	const unsigned long target_size )
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
		
	randomize_samples(images, objectLocations);

	std::cout << numFolds << "-fold cross validation (precision,recall,AP): "
		<< cross_validate_object_detection_trainer(trainer, images, objectLocations, ignore, numFolds) << std::endl;
}

void TrainFHog::train(std::vector<cv::Mat> gradientSamplesList, std::vector<int> labels)
{
	typedef dlib::matrix < float, 1980, 1 > sample_type;
	typedef dlib::radial_basis_kernel< sample_type > kernel_type;
	std::vector < sample_type > samples;
	std::vector < float > flLabels(labels.begin(), labels.end());
	dlib::svm_nu_trainer < kernel_type > trainer;

	for(auto &sample : gradientSamplesList)
	{
		dlib::cv_image<float> cvTmp(sample);
		dlib::matrix<float,1980,1> mtxTmp = dlib::mat(cvTmp);
		samples.push_back(mtxTmp);
	}

	std::cout << "All samples : " <<  samples.size() << std::endl;
				
	trainer.set_kernel(kernel_type(_gamma_par));
	trainer.set_nu(_nu_par);

	dlib::serialize(_namefile) <<  trainer.train(samples, flLabels);
}