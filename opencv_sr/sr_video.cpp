#include <opencv2/opencv.hpp>
#include "opencv2/superres.hpp"

using namespace cv;
using namespace std;
using namespace cv::superres;

int main(int argc, char** argv)
{

//    const string inputVideoName = "./data/lenica_rhinoceros_1965_512kb.mp4";
//    2009-09-08_marc-welz_pam_512kb.mp4
//     AccessMath_lecture_01_part_3.mp4

    CommandLineParser cmd(argc, argv,
        "{ v video      |           | Input video (mandatory)}"
    );

    string inputVideoName = cmd.get<string>("video");
    if ( inputVideoName.empty())
    {
        inputVideoName = "./data/lenica_rhinoceros_1965_512kb.mp4";
    }

    Ptr<cv::superres::FrameSource> f;
    f = cv::superres::createFrameSource_Video(inputVideoName);
    Ptr<cv::superres::SuperResolution >sr;

    VideoCapture vid;
    vid.open(0);

    sr = cv::superres::createSuperResolution_BTVL1();
    Ptr<DenseOpticalFlowExt> of = cv::superres::createOptFlow_Farneback();

    int scale = 2;
    sr->setOpticalFlow(of);
    sr->setScale(scale);
    sr->setTemporalAreaRadius(2);
    sr->setIterations(2);
    Mat frame;
    char c;
//    Mat frameZoom;
//    Mat mZoom;
//    Mat v;
    f->nextFrame(frame);
    sr->setInput(f);
    do{
        sr->nextFrame(frame);
        imshow("Video SuperRes", frame);
//        if (vid.grab())
//        {
//            vid >> v;
//            resize(v, mZoom, Size(0, 0), scale, scale);
//            imshow("Video Scale", mZoom);
//
//        }
//            imshow("Video SuperRes", frame);
        c = waitKey(20);

    } while (c != 27);
    return 0;
}