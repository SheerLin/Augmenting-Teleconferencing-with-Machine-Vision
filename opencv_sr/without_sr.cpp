#include <opencv2/opencv.hpp>
#include "opencv2/superres.hpp"

using namespace cv;
using namespace std;
using namespace cv::superres;

int main(int argc, char** argv)
{

    Ptr<cv::superres::FrameSource> f;
    f = cv::superres::createFrameSource_Camera(0);

    VideoCapture vid;
    vid.open(0);

    Mat frame;
    char c;
    f->nextFrame(frame);
    do{
        f->nextFrame(frame);
        imshow("No Video SuperRes", frame);
        c = waitKey(20);
    } while (c != 27);
    return 0;
}