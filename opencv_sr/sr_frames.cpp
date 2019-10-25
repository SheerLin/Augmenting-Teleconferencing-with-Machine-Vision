#include <opencv2/opencv.hpp>
#include "opencv2/superres.hpp"
#include <chrono> 

using namespace cv;
using namespace std;
using namespace cv::superres;


int main(int argc, char** argv)
{

    CommandLineParser cmd(argc, argv,
        "{ v video      |           | Input video (mandatory)}"
    );

    string inputVideoName = cmd.get<string>("video");
    if ( inputVideoName.empty())
    {
        inputVideoName = "./data/wb_mengmeng.mov";
    }

    Ptr<cv::superres::FrameSource> f1, f2;
    f1 = cv::superres::createFrameSource_Video(inputVideoName);
    f2 = cv::superres::createFrameSource_Video(inputVideoName);
    Ptr<cv::superres::SuperResolution >sr;

    VideoCapture vid;
    vid.open(0);

    sr = cv::superres::createSuperResolution_BTVL1();
    // sr = cv::superres::createSuperResolution_BTVL1_CUDA();
    Ptr<DenseOpticalFlowExt> of = cv::superres::createOptFlow_Farneback();

    int scale = 2;
    sr->setOpticalFlow(of);
    sr->setScale(scale);
    sr->setTemporalAreaRadius(2);
    sr->setIterations(2);
    Mat frame1;
    Mat frame2;
//    Mat frameZoom;
//    Mat mZoom;
//    Mat v;

    auto start = std::chrono::high_resolution_clock::now();
    sr->setInput(f2);
    int count = 0;
    do{
        f1->nextFrame(frame1);
        imwrite("frame" + std::to_string(count) + ".jpg", frame1);
       
        sr->nextFrame(frame2);
        imwrite("frame" + std::to_string(count) + "superres.jpg", frame2);
        
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Elapsed time: " << elapsed.count()  << "seconds, frame =" << count << "\n";
        count++;
//        if (vid.grab())
//        {
//            vid >> v;
//            resize(v, mZoom, Size(0, 0), scale, scale);
//            imshow("Video Scale", mZoom);
//
//        }
//            imshow("Video SuperRes", frame);
        // c = waitKey(20);// 

    } while (count < 10);
    return 0;
}