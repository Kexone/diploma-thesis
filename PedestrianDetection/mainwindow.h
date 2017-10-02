#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "mediafile.h"
#include "settings.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_buttonOpenVidImg_clicked();

    void on_buttonOpenWebcam_clicked();

    void on_buttonTrainPosSet_clicked();

    void on_buttonTrainNegSet_clicked();

    void on_buttonStartDetect_clicked();

    void on_comboBox_currentIndexChanged(int index);

private:
    Ui::MainWindow *ui;
    void appendBackLog(QString text);
    MediaFile *mediaFile;
    Settings settings;
    int totalFrames = 0;
    double startTime = 0;
    double endTime = 0;
    double fps = 0;
};

#endif // MAINWINDOW_H
