#include <SPDetector.hpp>
#include <Tools.hpp>

using namespace SuperPointSLAM;

/**  You need to modify the path below that corresponds to your dataset and weight path. **/
const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";
const std::string weight_dir = project_dir + "Weights/superpoint.pt";
const std::string dataset_dir = project_dir + "Dataset/";
const std::string kitti_dir = "/home/leecw/Datasets/kitti_gray_dataset/sequences/00/image_0/%06d.png";
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
    else ms = 10;
    std::cout << "Frame rate is " << ms << "ms.\n";
    

    /** Initialize VideoSteamer and SuperPoint Object **/ 
    // VideoStreamer vs("../Dataset/nyu_snippet.mp4");
    // VideoStreamer vs("/home/leecw/Datasets/Soongsil_Post/SoongsilMixed%4d.png");
    // VideoStreamer vs("/home/leecw/Datasets/Soongsil_Denoise10/p%5d.png");
    // VideoStreamer vs(0);
    VideoStreamer vs(kitti_dir);
    vs.setImageSize(cv::Size(720, 240));
    
    
    /** Superpoint Detector **/
    SPDetector SPF(weight_dir, torch::cuda::is_available());
    std::cout << "VC created, SPDetector Constructed.\n";

    // Test input/output file
    // std::ifstream inputFile("input.txt", std::ios::in);
    // std::ofstream outputFile("output.txt", std::ios::out | std::ios::app);
    // int test_nms_dist_thres;
    // float test_conf_thres;
    // inputFile >> test_nms_dist_thres >> test_conf_thres;


    cv::namedWindow("superpoint", cv::WINDOW_AUTOSIZE);
    long long idx=0;
    while(++idx){
        // Capture frame-by-frame
        // Image's size is [640 x 480]
        if(!vs.next_frame()) { std::cout << "main -- Video End\n"; break; }

        std::vector<cv::KeyPoint> Keypoints;
        cv::Mat Descriptors;

        auto start = std::chrono::system_clock::now();
        SPF.detect(vs.input, Keypoints, Descriptors);
        auto end = std::chrono::system_clock::now();
        std::chrono::milliseconds mill  = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        /* Logging */
        std::cout << idx << "th ProcessTime: " << mill.count() << "ms\n";
        std::cout << "Keypoint num: " << Keypoints.size() << std::endl;

        float x_scale(vs.W_scale), y_scale(vs.H_scale);

        auto kpt_iter = Keypoints.begin();
        for(; kpt_iter != Keypoints.end(); kpt_iter++)
        {
            float X(kpt_iter->pt.x), Y(kpt_iter->pt.y);
            double conf(kpt_iter->response);
            cv::circle(vs.img, cv::Point(int(X*x_scale), int(Y*y_scale)), 3, cv::Scalar(0, 0, (255 * conf * 10)), 2);
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