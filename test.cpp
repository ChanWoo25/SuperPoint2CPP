#include "test.hpp"

int main(const int argc, const char* argv[])
{
    using namespace NAMU_TEST;
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
    torch::load(model, "superpoint.pt");

    
    //auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, torch::kByte);

    cv::namedWindow("SuperPoint Tracker");
    auto font = cv::FONT_HERSHEY_DUPLEX;
    
    bool saving = true;
    if(saving)
    {
        string saving_dir = "/home/leecw/Reps/SuperPoint2CPP";  
    }
    string kDataRoot = "./data";
    

}