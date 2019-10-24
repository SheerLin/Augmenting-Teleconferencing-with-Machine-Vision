#include <opencv2/opencv.hpp> 
#include "opencv2/superres.hpp"

using namespace cv;
using namespace std;
using namespace cv::superres;

/* Reference:
 * https://answers.opencv.org/question/63946/super-resolution-video-format/
 */
int main(int argc, char** argv)
{

    Ptr<cv::superres::FrameSource> f;
    f = cv::superres::createFrameSource_Camera(0);
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
//        imshow("Video SuperRes", frame);
        c = waitKey(20);

    } while (c != 27);
    return 0;
}