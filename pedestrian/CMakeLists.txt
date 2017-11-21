cmake_minimum_required(VERSION 2.8.9)
project(Pedestrian)


#set(OPENCV "/media/56da5220-35b4-4144-9320-d10ca434d276_/opencv") #banana pi settings
#set(OPENCV "/home/debian/Downloads/opencv") #HummingBoard settings
#set(OPENCV_DIR "/media/56da5220-35b4-4144-9320-d10ca434d276_/opencv")
#find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui bgsegm)
set(CMAKE_CXX_COMPILER "g++-4.8") # Banana PI settings
#set(CMAKE_CXX_COMPILER "g++-4.9") #HummingBoard settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} - debug")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} - release")
include_directories(${OpenCV}/include)
link_directories(${OpenCV}/lib)

add_executable(Pedestrian main.cpp source/pipeline.cpp alg/convexhull.cpp alg/hog.cpp alg/mog.cpp alg/cascadeClass.cpp media/videostream.cpp media/mediafile.cpp train/trainhog.cpp test/svmTest.cpp )
target_link_libraries(Pedestrian opencv_core opencv_highgui opencv_bgsegm
opencv_imgproc opencv_video opencv_videoio opencv_objdetect)