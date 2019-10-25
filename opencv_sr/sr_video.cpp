#include <opencv2/opencv.hpp>
#include "opencv2/superres.hpp"
#include <chrono> 

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
        inputVideoName = "./data/wb_mengmeng.mov";
    }
    string outputVideoName = "superres.mp4";

    Ptr<cv::superres::FrameSource> f;
    f = cv::superres::createFrameSource_Video(inputVideoName);
    Ptr<cv::superres::SuperResolution >sr;
    VideoWriter writer;

    sr = cv::superres::createSuperResolution_BTVL1();
    Ptr<DenseOpticalFlowExt> of = cv::superres::createOptFlow_Farneback();

    int scale = 2;
    sr->setOpticalFlow(of);
    sr->setScale(scale);
    sr->setTemporalAreaRadius(2);
    sr->setIterations(2);
    Mat frame;
    char c;

    auto start = std::chrono::high_resolution_clock::now();

    sr->setInput(f);
    int count = 0;
    while(true){
        sr->nextFrame(frame);
        if (count > 10)
            break;
        if (!outputVideoName.empty())
        {
            std::cout<<"1";
            if (!writer.isOpened()) {
                std::cout<<"2";

                // https://www.fourcc.org/mp4v/   => mpg-4
                writer.open(outputVideoName, VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, frame.size());
                std::cout<<"3";
            }
            writer << frame; 
            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;
            std::cout << "Elapsed time: " << elapsed.count()  << "seconds, frame =" << count << "\n";
            count++;
        }
    }
    writer.release();
    return 0;
}