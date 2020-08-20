#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <ctime>
#include "test.hpp"

const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";

int main(const int argc, const char* argv[])
{
    using namespace NAMU_TEST;
    NAMU_TEST::Explain = true;

    // std::cout << *typeid(float(CV_PI)).name() << std::endl;
    // Output : f

    std::cout << "argc : " << argc << std::endl;
    for(int i=0 ;i<argc; i++){
        std::cout << argv[i] << std::endl;
    }
    /*************************************************************/
    printSection(1, "Cuda availablility.");
    bool use_cuda = torch::cuda::is_available();
    //display();

    /*************************************************************/
    printSection(2, "Module Information.");
    
    //display(net);

    

    /*************************************************************/
    printSection(3, "Load");
    DeviceType device_type;
    device_type = (use_cuda) ? kCUDA : kCPU;
    std::cout << "Device type is " << device_type << std::endl;
    Device device(device_type);

    std::shared_ptr<SuperPoint> model;
    model = std::make_shared<SuperPoint>();
    std::cout << "model constructor.\n";

    load(model, project_dir + "Weights/superpoint.pt");
    model->to(device);


    auto TsOpt = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(device)
                .requires_grad(false);

    Tensor input = torch::rand({1, 1, 64,64}, TsOpt) * 256;
    //std::cout << input << std::endl;
    
    auto output = model->forward(input);

    if(use_cuda)
        model->to(torch::kCPU);

    std::cout << "size is " << output.size() << std::endl;
    output[0].print();
    // auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);

    // cv::namedWindow("SuperPoint Tracker");
    // auto font = cv::FONT_HERSHEY_DUPLEX;

    // bool saving = true;
    // if(saving)
    // {
    //     string saving_dir = "/home/leecw/Reps/SuperPoint2CPP";  
    // }
    // string kDataRoot = "./data";
}