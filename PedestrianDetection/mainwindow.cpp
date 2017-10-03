#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "videostream.h"
#include <QFileDialog>
#include <QStringList>
#include <QString>

int MainWindow::totalFrames = 0;
float MainWindow::fps  = 0;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->textBackLog->clear();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setTotalFrames(int allFrames)
{
    MainWindow::totalFrames = allFrames;
}

void MainWindow::setFps(float nFps)
{
    MainWindow::fps = nFps;
}

void MainWindow::on_buttonOpenVidImg_clicked()
{
    QString text = "Error in loading";
    QStringList fileName = QFileDialog::getOpenFileNames(this,
         tr("Open Video/Image"), "", tr("Supported Files (*.png *.jpg *.bmp *.pgm *.seq *.avi *.mp4)"));
    if(fileName[0].contains(".mp4") || fileName[0].contains(".avi") || fileName[0].contains(".seq")) {
        videoStream = new VideoStream();
        isVideo = true;
        text = QString::fromUtf8(videoStream->openFile(fileName[0].toUtf8().constData()).c_str());
    }
    else {
        isVideo = false;
        mediaFile = new MediaFile(fileName);
        text = QString::number(fileName.size()) + " file(s) loaded.";
    }
    appendBackLog(text);
}

void MainWindow::on_buttonOpenWebcam_clicked()
{

}

void MainWindow::on_buttonTrainPosSet_clicked()
{

}

void MainWindow::on_buttonTrainNegSet_clicked()
{

}

void MainWindow::on_buttonStartDetect_clicked()
{
    startTime = (double)cv::getTickCount();
    settings.hogThreshold = ui->inputHogTresh->text().toFloat();
    settings.mogThreshold = ui->inputMogTresh->text().toFloat();
    settings.algorithm = ui->comboBoxTypeAlg->currentIndex();
    settings.convexHullSize = cv::Size(ui->inputCHSizeMin->text().toFloat(), ui->inputCHSizeMax->text().toFloat());
    settings.convexHUllDeviation = cv::Size(ui->inputCHDevMin->text().toFloat(), ui->inputCHDevMax->text().toFloat());
    settings.showVideoFrames = ui->checkBoxShowVideoFrames->isChecked();
    settings.positiveFrames = ui->inputOthPosFrames->text().toInt();
    //settings.trainHog = ui->;
    appendBackLog("START Detection");

    if(isVideo) {
        appendBackLog("VIDEO");
        pipeline.chooseType(1, videoStream->getFrames());
    }
    else {
        appendBackLog("IMAGE");
        pipeline.chooseType(1, mediaFile->getFrames());
    }
    appendBackLog(QString::number(settings.hogThreshold));
    appendBackLog(QString::number(settings.mogThreshold));
    appendBackLog(QString::number(settings.algorithm));
    appendBackLog(QString::number(settings.positiveFrames));

    endTime = (double)cv::getTickCount() - startTime;
    double totalTime = roundf((endTime / cv::getTickFrequency())*100)/100;
    appendBackLog("Total frames: " + QString::number(MainWindow::totalFrames));
    appendBackLog("Time: " + QString::number(totalTime));
    appendBackLog("FPS : " + QString::number(MainWindow::fps));
    appendBackLog("Video duration: " + QString::number(MainWindow::totalFrames / MainWindow::fps));
    appendBackLog("DONE");



}

void MainWindow::appendBackLog(QString text)
{
    ui->textBackLog->append(text);
}
