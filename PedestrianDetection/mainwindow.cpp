#include "mainwindow.h"
#include "ui_mainwindow.h"
//#include "videostream.h"
#include <QFileDialog>
#include <QStringList>
#include <QString>
#include <QThread>

int MainWindow::totalFrames = 0;
float MainWindow::fps  = 0;
double Settings::hogThreshold = 0;
double Settings::mogThreshold = 0;
int Settings::algorithm = 0;
cv::Size Settings::convexHullSize = cv::Size(0, 0);
cv::Size Settings::convexHUllDeviation = cv::Size(0, 0);
bool Settings::showVideoFrames = false;
int Settings::positiveFrames = 0;

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
    QStringList files = QFileDialog::getOpenFileNames(this,
         tr("Open Video/Image"), "", tr("Supported Files (*.png *.jpg *.bmp *.pgm *.seq *.avi *.mp4)"));
   // std::vector<std::string> fileName{std::begin(files), std::end(files)};
    std::vector<std::string> fileName = convertQstring(files);
    std::string suffix = fileName[0].substr(fileName[0].find_last_of(".") + 1);
    if(suffix == "mp4" || suffix == ".avi" || suffix == ".seq") {
        this->thread()->sleep(1);
        isVideo = true;
        mediaFile = new MediaFile(isVideo);
        text = QString::fromStdString(mediaFile->openFile(fileName));
    }
    else {
        this->thread()->sleep(1);
        isVideo = false;
        mediaFile = new MediaFile(isVideo);
        text = QString::fromStdString(mediaFile->openFile(fileName));
    }
    fileName.clear();
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
    Settings::hogThreshold = ui->inputHogTresh->text().toDouble();
    Settings::mogThreshold = ui->inputMogTresh->text().toDouble();
    Settings::algorithm = ui->comboBoxTypeAlg->currentIndex();
    Settings::convexHullSize = cv::Size(ui->inputCHSizeMin->text().toDouble(), ui->inputCHSizeMax->text().toDouble());
    Settings::convexHUllDeviation = cv::Size(ui->inputCHDevMin->text().toDouble(), ui->inputCHDevMax->text().toDouble());
    Settings::showVideoFrames = ui->checkBoxShowVideoFrames->isChecked();
    Settings::positiveFrames = ui->inputOthPosFrames->text().toInt();
    //settings.trainHog = ui->;
    appendBackLog("START Detection");

    if(isVideo) {
        appendBackLog("VIDEO");
        pipeline.chooseType(1, mediaFile->getFrames());
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

std::vector<std::string> MainWindow::convertQstring(QStringList files)
{
    std::vector<std::string> temp;
    for(QString file : files) {
        temp.emplace_back(file.toUtf8().constData());
    }
    return temp;
}
