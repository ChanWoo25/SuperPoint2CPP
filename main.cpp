#include <SPDetector.hpp>
#include <Tools.hpp>

using namespace SuperPointSLAM;

const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";
const std::string weight_dir = project_dir + "Weights/superpoint.pt";
const std::string dataset_dir = project_dir + "Dataset/";
void test();


int main(const int argc, char* argv[])
{
    /** From the main argument, Retrieve waiting period to control Displaying.**/
    int ms;
    if(argc == 2)
    {   
        char* a = argv[1];
        ms = std::atoi(a);
    }
    else ms = 100;
    std::cout << "Frame rate is " << ms << "ms.\n";
    

    /** Initialize VideoSteamer and SuperPoint Object **/ 
    // VideoStreamer vs("../Dataset/nyu_snippet.mp4");
    VideoStreamer vs(0);
    SPDetector SPF(weight_dir, torch::cuda::is_available());
    std::cout << "VC created, SPDetector Constructed.\n";

    std::ifstream inputFile("input.txt", std::ios::in);
    std::ofstream outputFile("output.txt", std::ios::out | std::ios::app);
    int test_nms_dist_thres;
    float test_conf_thres;
    inputFile >> test_nms_dist_thres >> test_conf_thres;


    cv::namedWindow("superpoint", cv::WINDOW_AUTOSIZE);
    long long idx=0;
    while(++idx){
        // Capture frame-by-frame
        // Image's size is [640 x 480]
        if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }


        auto start = std::chrono::system_clock::now();
        auto descriptors = SPF.detect(vs.input);
        auto end = std::chrono::system_clock::now();
        std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        
        //SPF.computeDescriptors(desciptors);
        std::cout << idx << "-th n_keypoint: " << SPF.kpts_nms.size()
                << " - Processing time: " << mill.count() << "ms\n";
        float x_scale(vs.W_scale), y_scale(vs.H_scale);
        for(auto iter = SPF.kpts_nms.begin(); iter != SPF.kpts_nms.end(); iter++)
        {
            float X((*iter).pt.x), Y((*iter).pt.y);
            cv::circle(vs.img, cv::Point(int(X*x_scale), int(Y*y_scale)), 3, cv::Scalar(0, 0, 255), 2);
        }

        // Display the resulting frame
        cv::imshow( "superpoint", vs.img );

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(ms);
        if(c==27){ break; }
    }


    // Closes all the frames
    cv::destroyAllWindows();
}