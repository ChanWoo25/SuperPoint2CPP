#include "test.hpp"

const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";
const std::string weight_dir = project_dir + "Weights/superpoint.pt";

int main(const int argc, const char* argv[])
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

    auto tensor_options = torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .layout(torch::kStrided)
                            .device(c10::DeviceType(torch::kCUDA))
                            .requires_grad(false);

    cv::Mat cv_mat = cv::Mat::eye(3,3,CV_32F);
    torch::Tensor tensor = torch::zeros({3, 3}, torch::kF32);

    std::memcpy(tensor.data_ptr(), cv_mat.data, sizeof(float)*tensor.numel());

    std::cout << cv_mat << std::endl;
    std::cout << tensor << std::endl;

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



}


    // std::shared_ptr<SuperPoint> model;
    // model = std::make_shared<SuperPoint>();
    // std::cout << "model constructor.\n";
    // load(model, project_dir + "Weights/superpoint.pt");
    //  model->to(device);