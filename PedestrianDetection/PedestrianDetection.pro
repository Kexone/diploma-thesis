#-------------------------------------------------
#
# Project created by QtCreator 2017-09-24T15:25:09
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = PedestrianDetection
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    main.cpp \
    mainwindow.cpp \
    convexhull.cpp \
    hog.cpp \
    mog.cpp \
    videostream.cpp \
    mediafile.cpp

HEADERS  += mainwindow.h \
    mainwindow.h \
    convexhull.h \
    hog.h \
    mog.h \
    videostream.h \
    mediafile.h \
    settings.h

FORMS    += mainwindow.ui

DISTFILES += \
    icons/if_cancel-2_309095.png \
    icons/if_pen_stroke_sketch_doodle_lineart_72_451308.png \
    icons/non-pedestrian.png \
    icons/noun_42436.png \
    icons/pedestrian.png \
    icons/webcam.png \
    icons/if_cancel-2_309095.ico \
    icons/if_pen_stroke_sketch_doodle_lineart_72_451308.ico \
    icons/webcam.ico

win32:CONFIG(release, debug|release): LIBS += -L$$PWD${OpenCV_dir}/lib/ -lopencv_core -lopencv_highgui -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_objdetect
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../../../OpenCV/lib/ -lopencv_core320d -lopencv_highgui320d -lopencv_bgsegm320d -lopencv_imgproc320d -lopencv_video320d -lopencv_videoio320d -lopencv_imgcodecs320d -lopencv_objdetect320d

INCLUDEPATH += $$PWD/../../../../../../OpenCV/include
DEPENDPATH += $$PWD/../../../../../../OpenCV/include
