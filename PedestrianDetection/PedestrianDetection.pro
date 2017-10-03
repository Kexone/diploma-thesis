#-------------------------------------------------
#
# Project created by QtCreator 2017-09-24T15:25:09
#
#-------------------------------------------------

QT       += core gui

CONFIG += c++11

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PedestrianDetection
TEMPLATE = app
SOURCES += main.cpp\
        mainwindow.cpp \
    convexhull.cpp \
    hog.cpp \
    mog.cpp \
    videostream.cpp \
    mediafile.cpp \
    pipeline.cpp

HEADERS  += mainwindow.h \
    convexhull.h \
    hog.h \
    mog.h \
    videostream.h \
    mediafile.h \
    settings.h \
    pipeline.h

FORMS    += mainwindow.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD${OpenCV_dir}/lib/ -lopencv_core -lopencv_highgui -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_objdetect
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../OpenCV/lib/ -lopencv_core320d -lopencv_highgui320d -lopencv_bgsegm320d -lopencv_imgproc320d -lopencv_video320d -lopencv_videoio320d -lopencv_imgcodecs320d -lopencv_objdetect320d
unix:!macx:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../openCV/debug/lib/ -lopencv_core -lopencv_highgui -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_objdetect
else:unix:!macx:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../openCV/release/lib/ -lopencv_core -lopencv_highgui -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_objdetect

win32:INCLUDEPATH += $$PWD/../../../../../../OpenCV/include
win32:DEPENDPATH += $$PWD/../../../../../../OpenCV/include


unix:!macx:INCLUDEPATH += $$PWD/../../../openCV/debug/include
unix:!macx:DEPENDPATH += $$PWD/../../../openCV/debug/include
