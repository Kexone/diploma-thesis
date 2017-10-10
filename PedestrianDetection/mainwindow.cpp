#include "mainwindow.h"
#include "ui_mainwindow.h"
//#include "videostream.h"
#include <QFileDialog>
#include <QInputDialog>
#include <QStringList>
#include <QString>
#include <QThread>
#include <QCameraInfo>

int MainWindow::totalFrames = 0;
float MainWindow::fps  = 0;
double Settings::hogThreshold = 0;
double Settings::mogThreshold = 0;
int Settings::mogHistory = 0;
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
    if(!files.empty()) {
    std::vector<std::string> fileName = convertQstring(files);
    std::string suffix = fileName[0].substr(fileName[0].find_last_of(".") + 1);
    if(suffix == "mp4" || suffix == ".avi" || suffix == ".seq") {
        this->thread()->sleep(1);
        isVideo = true;
        fileFeed = fileName[0];
        text = QString::fromStdString(fileName[0] + " loaded.");
    }
    else {
        this->thread()->sleep(1);
        isVideo = false;
        mediaFile = new MediaFile();
        text = QString::fromStdString(mediaFile->openFile(fileName));
    }
    fileName.clear();
    appendBackLog(text);
    }
}

void MainWindow::on_buttonOpenWebcam_clicked()
{
    QStringList webcams;
    const QList<QCameraInfo> cameras = QCameraInfo::availableCameras();
    for(QCameraInfo cam : cameras) {
        webcams.push_back(cam.description() + " (" + cam.deviceName() + ")");
    }
    bool ok;
    QString text = QInputDialog::getItem(this, tr("Select video stream"),
                                         tr("Webcam:"),webcams,0,false, &ok);
    if (ok && !text.isEmpty()) {
        for(int i = 0; i < webcams.size(); i++) {
            if(QString::compare(text,webcams[i],Qt::CaseInsensitive) == 0 ){
                this->cameraFeed = i;
                break;
            }
        }
        appendBackLog("Webcam selected: " + text.mid(0,text.indexOf("(")));
    }
    webcams.clear();
    text.clear();
}

void MainWindow::on_buttonTrainPosSet_clicked()
{
    this->thread()->exit();
}

void MainWindow::on_buttonTrainNegSet_clicked()
{

}

void MainWindow::on_buttonStartDetect_clicked()
{
    setSettings();
    appendBackLog("START Detection");
    startTime = (double)cv::getTickCount();
    if(cameraFeed != 99) {
        appendBackLog("WEBCAM");
        pipeline.execute(cameraFeed);
    }
    else if(isVideo) {
        appendBackLog("VIDEO");
        pipeline.execute(fileFeed);
    }
    else {
        appendBackLog("IMAGE");
        pipeline.execute(mediaFile->getFrames());
    }
    report();
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

void MainWindow::setSettings()
{
    Settings::hogThreshold = ui->inputHogTresh->text().toDouble();
    Settings::mogThreshold = ui->inputMogTresh->text().toDouble();
    Settings::mogHistory = ui->inputMogHistory->text().toInt();
    Settings::algorithm = ui->comboBoxTypeAlg->currentIndex();
    Settings::convexHullSize = cv::Size(ui->inputCHSizeMin->text().toDouble(), ui->inputCHSizeMax->text().toDouble());
    Settings::convexHUllDeviation = cv::Size(ui->inputCHDevMin->text().toDouble(), ui->inputCHDevMax->text().toDouble());
    Settings::showVideoFrames = ui->checkBoxShowVideoFrames->isChecked();
    Settings::positiveFrames = ui->inputOthPosFrames->text().toInt();
    //settings.trainHog = ui->;
}

void MainWindow::report()
{
    // DEBUG
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
