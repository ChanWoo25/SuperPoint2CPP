#include "test.hpp"

const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";
const std::string weight_dir = project_dir + "Weights/superpoint.pt";
const std::string dataset_dir = project_dir + "Dataset/";

int main(const int argc, char* argv[])
{
    using namespace NAMU_TEST;

    /*************************************************************/
    printSection(1, "Cuda availablility. [DONE]");
    bool use_cuda = torch::cuda::is_available();
    //display();

    /*************************************************************/
    printSection(2, "Module Information. [DONE]");
    //display(net);

    /*************************************************************/
    printSection(3, "Load Weight. [DONE]");

    /*************************************************************/
    printSection(3, "Understand Forward procedure. [DONE]");

    // auto tensor_options = torch::TensorOptions()
    //                         .dtype(torch::kFloat32)
    //                         .layout(torch::kStrided)
    //                         .device(c10::DeviceType(torch::kCUDA))
    //                         .requires_grad(false);

    // cv::Mat cv_mat = cv::Mat::eye(3,3,CV_32F);
    // torch::Tensor tensor = torch::zeros({3, 3}, torch::kF32);

    // std::memcpy(tensor.data_ptr(), cv_mat.data, sizeof(float)*tensor.numel());

    // std::cout << cv_mat << std::endl;
    // std::cout << tensor << std::endl;

    // cv::Mat key(5 , 2, CV_32FC1, 1);
    // cv::Mat val(5 , 1, CV_32FC1, 2);

    // std::cout << key << std::endl;
    // std::cout << val << std::endl;

    // auto xy = key.ptr<float>(0);
    // auto val_ptr = val.ptr<float>(0);

    // for (size_t i = 0; i < 5; i++)
    // {
    //     //auto xy = key.ptr<float>(i);
    //     int x = *(xy++) = float(i * 2 + 0);
    //     int y = *(xy++) = float(i * 2 + 1);
    //     *(val_ptr++) = float(x * 10 + y);
    // }
    // std::cout << key << std::endl;
    // std::cout << val << std::endl;

    // std::vector<cv::KeyPoint> kpts;
    // kpts.push_back(cv::KeyPoint(cv::Point2d(1, 2), 8, -1, 1.8f));
    // kpts.push_back(cv::KeyPoint(cv::Point2d(7, 3), 8, -1, 12.1f));
    // kpts.push_back(cv::KeyPoint(cv::Point2d(3, 4), 8, -1, 7.4f));
    // kpts.push_back(cv::KeyPoint(cv::Point2d(5, 1), 8, -1, 4.6f));
    // std::sort(kpts.begin(), kpts.end(), 
    //     [](cv::KeyPoint a, cv::KeyPoint b) { return a.response > b.response; });
    // for(auto kpt: kpts)
    //     std::cout << kpt.pt << ", " << kpt.response << std::endl;
    
    
    /*************************************************************/
    printSection(5, "cv::Mat's ROI [DONE]");
    // cv::Mat mat(cv::Size(4, 4), CV_32FC1, cv::Scalar(1));
    // std::cout << mat << std::endl; 

    // cv::Mat sub = cv::Mat(mat, cv::Rect(cv::Point2d(1, 1), cv::Point2d(3, 3)));
    // sub.setTo(0);
    // std::cout << mat << std::endl;

    printSection(6, "read image. [DONE] ");
    //test_with_magicleap();
    

    printSection(7, "read video. [DONE] ");


    // cv::Mat m1(3, 3, CV_32FC1), m2(3, 3, CV_32FC1);
    // cv::randn(m1, 10, 5);
    // cv::randn(m2, 10, 5);

    // std::cout << m1.t() << std::endl << m2 << std::endl;
    // //std::cout << m1.dot(m2) << std::endl;
    // try { 
    //     std::cout << (m1.t() * m2) << std::endl; 
    // } 
    // catch (cv::Exception const & e) { 
    //     std::cerr<<"OpenCV exception: "<<e.what()<<std::endl; 
    //     }
    
    // std::cout << cv::max(m1, m2) << std::endl;
    
    printSection(8, "CV type check. [DONE] ");
    // int A[] = {1, 2, 3, 4, 5, 6};
    // cv::Mat a(2, 3, CV_32S, A);
    // std::cout << a << "\n\n";

    // cv::Mat b = (a <= 3);
    // std::cout << b << "\n\n";
    // std::cout << (b.type() == CV_8U) << "\n\n";
    
    int ms;
    if(argc == 2)
    {   
        char* a = argv[1];
        ms = std::atoi(a);
    }
    else ms = 200;
    std::cout << "Frame rate is " << ms << "ms.\n";
    
    cv::namedWindow("superpoint", cv::WINDOW_AUTOSIZE);

    //VideoStreamer vs("../Dataset/nyu_snippet.mp4");
    VideoStreamer vs(0);
    SuperPointFrontend SPF(weight_dir, torch::cuda::is_available());
    std::cout << "VC created, SuperpointFrontend Constructed.\n";
    
    int idx=1;
    while(idx++){
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


    // std::shared_ptr<SuperPoint> model;
    // model = std::make_shared<SuperPoint>();
    // std::cout << "model constructor.\n";
    // load(model, project_dir + "Weights/superpoint.pt");
    //  model->to(device);