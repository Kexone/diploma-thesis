#ifndef TRAINFHOG_H
#define TRAINFHOG_H
#include <iostream>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>
#include <dlib/pixel.h>
#include <dlib/opencv/cv_image.h>

class TrainFHog
{
public: 
	using pixel_type = dlib::bgr_pixel;

	TrainFHog();

	/**
	* @brief
	*
	* @param
	*/
	void train(std::string posSamples, std::string negSamples);
protected:

	/**
	* @brief
	*
	* @param
	*/
	void testParams(std::vector< cv::Mat > gradientSamplesList, std::vector< int > labels);
private:

	/**
	* @brief
	*
	* @param
	*/
	bool containsAnyBoxes(const std::vector<std::vector<dlib::rectangle> >& boxes);

	/**
	* @brief
	*
	* @param
	*/
	void pickBestWindowSize(const std::vector<std::vector<dlib::rectangle> >& boxes, unsigned long& width, unsigned long& height, const unsigned long target_size);

	/**
	* @brief
	*
	* @param
	*/
	void throwInvalidBoxErrorMessage(const std::string& dataset_filename, const std::vector<std::vector<dlib::rectangle> >& removed, const unsigned long target_size);
};

#endif // TRAINFHOG_H

// ----------------------------------------------------------------------------------------

//class very_simple_feature_extractor : dlib::noncopyable
//{
//	/*!
//	WHAT THIS OBJECT REPRESENTS
//	This object is a feature extractor which goes to every pixel in an image and
//	produces a 32 dimensional feature vector.  This vector is an indicator vector
//	which records the pattern of pixel values in a 4-connected region.  So it should
//	be able to distinguish basic things like whether or not a location falls on the
//	corner of a white box, on an edge, in the middle, etc.
//
//
//	Note that this object also implements the interface defined in dlib/image_keypoint/hashed_feature_image_abstract.h.
//	This means all the member functions in this object are supposed to behave as
//	described in the hashed_feature_image specification.  So when you define your own
//	feature extractor objects you should probably refer yourself to that documentation
//	in addition to reading this example program.
//	!*/
//
//
//public:
//
//	template <
//		typename image_type
//	>
//		inline void load(
//			const image_type& img
//		)
//	{
//		feat_image.set_size(img.nr(), img.nc());
//		assign_all_pixels(feat_image, 0);
//		for (long r = 1; r + 1 < img.nr(); ++r)
//		{
//			for (long c = 1; c + 1 < img.nc(); ++c)
//			{
//				unsigned char f = 0;
//				if (img[r][c])   f |= 0x1;
//				if (img[r][c + 1]) f |= 0x2;
//				if (img[r][c - 1]) f |= 0x4;
//				if (img[r + 1][c]) f |= 0x8;
//				if (img[r - 1][c]) f |= 0x10;
//
//				// Store the code value for the pattern of pixel values in the 4-connected
//				// neighborhood around this row and column.
//				feat_image[r][c] = f;
//			}
//		}
//	}
//
//	inline unsigned long size() const { return feat_image.size(); }
//	inline long nr() const { return feat_image.nr(); }
//	inline long nc() const { return feat_image.nc(); }
//
//	inline long get_num_dimensions(
//	) const
//	{
//		// Return the dimensionality of the vectors produced by operator()
//		return 32;
//	}
//
//	typedef std::vector<std::pair<unsigned int, double> > descriptor_type;
//
//	inline const descriptor_type& operator() (
//		long row,
//		long col
//		) const
//		/*!
//		requires
//		- 0 <= row < nr()
//		- 0 <= col < nc()
//		ensures
//		- returns a sparse vector which describes the image at the given row and column.
//		In particular, this is a vector that is 0 everywhere except for one element.
//		!*/
//	{
//		feat.clear();
//		const unsigned long only_nonzero_element_index = feat_image[row][col];
//		feat.push_back(std::make_pair(only_nonzero_element_index, 1.0));
//		return feat;
//	}
//
//	// This block of functions is meant to provide a way to map between the row/col space taken by
//	// this object's operator() function and the images supplied to load().  In this example it's trivial.  
//	// However, in general, you might create feature extractors which don't perform extraction at every 
//	// possible image location (e.g. the hog_image) and thus result in some more complex mapping.  
//	inline const dlib::rectangle get_block_rect(long row, long col) const { return dlib::centered_rect(col, row, 3, 3); }
//	inline const dlib::point image_to_feat_space(const dlib::point& p) const { return p; }
//	inline const dlib::rectangle image_to_feat_space(const dlib::rectangle& rect) const { return rect; }
//	inline const dlib::point feat_to_image_space(const dlib::point& p) const { return p; }
//	inline const dlib::rectangle feat_to_image_space(const dlib::rectangle& rect) const { return rect; }
//
//	inline friend void serialize(const very_simple_feature_extractor& item, std::ostream& out) { serialize(item.feat_image, out); }
//	inline friend void deserialize(very_simple_feature_extractor& item, std::istream& in) { deserialize(item.feat_image, in); }
//
//	void copy_configuration(const very_simple_feature_extractor& item) {}
//
//private:
//	dlib::array2d<unsigned char> feat_image;
//
//	// This variable doesn't logically contribute to the state of this object.  It is here
//	// only to avoid returning a descriptor_type object by value inside the operator() method.
//	mutable descriptor_type feat;
//};
