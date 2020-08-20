#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <ctime>
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

 

    cv::Mat key(5 , 2, CV_32FC1, 1);
    cv::Mat val(5 , 1, CV_32FC1, 2);

    std::cout << key << std::endl;
    std::cout << val << std::endl;

    auto xy = key.ptr<float>(0);
    auto val_ptr = val.ptr<float>(0);

    for (size_t i = 0; i < 5; i++)
    {
        //auto xy = key.ptr<float>(i);
        int x = *(xy++) = float(i * 2 + 0);
        int y = *(xy++) = float(i * 2 + 1);
        *(val_ptr++) = float(x * 10 + y);
    }
    std::cout << key << std::endl;
    std::cout << val << std::endl;

}


    // std::shared_ptr<SuperPoint> model;
    // model = std::make_shared<SuperPoint>();
    // std::cout << "model constructor.\n";
    // load(model, project_dir + "Weights/superpoint.pt");
    //  model->to(device);