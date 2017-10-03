#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "mediafile.h"
#include "videostream.h"
#include "settings.h"
#include "pipeline.h"
namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    static void setFps(float nFps);
    static void setTotalFrames(int allFrames);

private slots:
    void on_buttonOpenVidImg_clicked();

    void on_buttonOpenWebcam_clicked();

    void on_buttonTrainPosSet_clicked();

    void on_buttonTrainNegSet_clicked();

    void on_buttonStartDetect_clicked();

private:
    bool isVideo = false;
    Ui::MainWindow *ui;
    void appendBackLog(QString text);
    MediaFile *mediaFile;
    VideoStream *videoStream;
    Settings settings;
    static int totalFrames;
    double startTime = 0;
    double endTime = 0;
    static float fps;
    Pipeline pipeline;
};

#endif // MAINWINDOW_H
