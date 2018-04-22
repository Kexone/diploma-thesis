cmake_minimum_required(VERSION 3.0.2)
project(Pedestrian)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -pedantic -pthread -ljpeg -lpng -lX11 -lblas -llapack")

	set(CMAKE_CXX_COMPILER "g++-4.9")
if(${arm} MATCHES "rpi") 
	set(OpenCV "/home/pi/Downloads/ocv_build")
	set(dlib "/home/pi/Downloads/dlib_build")
elseif(${arm} MATCHES "bpi") 
	set(OpenCV "/root/Downloads/ocv_build")
	set(dlib "/root/Downloads/dlib_build")
elseif(${arm} MATCHES "imx") 
	set(OpenCV "/home/debian/Downloads/ocv_build")
	set(dlib "/home/debian/Downloads/dlib_build")
else(${arm}) 
	message("ERROR with parsing configuration\n bpi - Banana PI \n rpi - Raspberry PI \n imx - Humming Board")
endif (${arm} MATCHES "rpi")


include_directories(${OpenCV}/include ${dlib})
link_directories(${OpenCV}/lib ${dlib}/lib)

add_executable(Pedestrian main.cpp
	source/pipeline.cpp source/testingPipeline.cpp source/settings.h
	source/alg/convexhull.cpp source/alg/hog.cpp source/alg/fhog.cpp source/alg/mog.cpp source/alg/cascadeClass.cpp
	source/media/videostream.cpp
	source/train/trainhog.cpp source/train/trainfhog.cpp source/train/trainCascade.cpp source/train/combinedTrainHog.cpp
	source/test/diffEvoTest.cpp source/test/testClass.cpp source/test/svmTest.cpp source/test/dlibSvmTest.cpp source/3dparty/de/DifferentialEvolution.h
	source/utils/extractorROI.cpp source/utils/utils.h
	)
target_link_libraries(Pedestrian libdlib.a opencv_core opencv_highgui opencv_bgsegm opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio opencv_objdetect opencv_ml)
