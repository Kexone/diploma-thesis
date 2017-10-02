#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "videostream.h"
#include <QFileDialog>
#include <QStringList>
#include <QString>


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

void MainWindow::on_buttonOpenVidImg_clicked()
{
    QStringList fileName = QFileDialog::getOpenFileNames(this,
         tr("Open Video/Image"), "", tr("Supported Files (*.png *.jpg *.bmp *pgm *seq)"));
    mediaFile = new MediaFile(fileName);
    appendBackLog(QString::number(fileName.size()) + " file(s) loaded.");
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
    appendBackLog(QString::number(settings.hogThreshold));
    appendBackLog(QString::number(settings.mogThreshold));
    appendBackLog(QString::number(settings.algorithm));
    appendBackLog(QString::number(settings.positiveFrames));

    endTime = (double)cv::getTickCount() - startTime;
    double totalTime = roundf((endTime / cv::getTickFrequency())*100)/100;
    appendBackLog("Total frames: " + totalFrames);
    appendBackLog("Time: " + QString::number(totalTime));
    appendBackLog("FPS : " + QString::number(fps));



}

void MainWindow::on_comboBox_currentIndexChanged(int index)
{
    appendBackLog("Selected alg: " + QString::number(index));
}

void MainWindow::appendBackLog(QString text)
{
    ui->textBackLog->append(text);
}
