#ifndef TEST_H
#define TEST_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <iostream>

namespace NAMU_TEST
{

using namespace torch;
using namespace nn;
bool Explain = true;

void printSection(int n, std::string s)
{
    std::cout << "\n < TEST Section " << n << " > \n" <<
                 "--Test about " << s << "--\n\n";
}


class SuperPoint : public Module {
// class를 쓰면, 권한을 지정해야한다. 상속시 public Module을 바꿨고,
// 멤버함수, 변수들도 public, private지정을 안해주면 모두 private처리가 되기 때문에 forward에 접할 수 없게된다.
public:
    SuperPoint()//생성자
    {

    /* 
    A Module is registered as a submodule to another Module 
    by calling register_module(), typically from within a parent 
    module’s constructor.
    */

    //SHARED ENCODER
        conv1a = register_module("conv1a", Conv2d(Conv2dOptions( 1, c1, 3).stride(1).padding(1)));
        conv1b = register_module("conv1b", Conv2d(Conv2dOptions(c1, c1, 3).stride(1).padding(1)));
        conv2a = register_module("conv2a", Conv2d(Conv2dOptions(c1, c2, 3).stride(1).padding(1)));
        conv2b = register_module("conv2b", Conv2d(Conv2dOptions(c2, c2, 3).stride(1).padding(1)));
        conv3a = register_module("conv3a", Conv2d(Conv2dOptions(c2, c3, 3).stride(1).padding(1)));
        conv3b = register_module("conv3b", Conv2d(Conv2dOptions(c3, c3, 3).stride(1).padding(1)));
        conv4a = register_module("conv4a", Conv2d(Conv2dOptions(c3, c4, 3).stride(1).padding(1)));
        conv4b = register_module("conv4b", Conv2d(Conv2dOptions(c4, c4, 3).stride(1).padding(1)));
    //DETECTOR
        convPa = register_module("convPa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
        convPb = register_module("convPb", Conv2d(Conv2dOptions(c5, 65, 3).stride(1).padding(0)));

    //DESCRIPTOR
        convDa = register_module("convDa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
        convDb = register_module("convDb", Conv2d(Conv2dOptions(c5, d1, 1).stride(1).padding(0)));
    }

    std::vector<torch::Tensor> forward(Tensor input); //순전파

private:
    //SHARED ENCODER
    Conv2d conv1a{nullptr};
    Conv2d conv1b{nullptr};
    Conv2d conv2a{nullptr};
    Conv2d conv2b{nullptr};
    Conv2d conv3a{nullptr};
    Conv2d conv3b{nullptr};
    Conv2d conv4a{nullptr};
    Conv2d conv4b{nullptr};

    //DETECTOR
    Conv2d convPa{nullptr};
    Conv2d convPb{nullptr};

    //DESCRIPTOR
    Conv2d convDa{nullptr};
    Conv2d convDb{nullptr};

    const int c1 = 64;
    const int c2 = 64;
    const int c3 = 128;
    const int c4 = 128;
    const int c5 = 256;
    const int d1 = 256;
};

cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms);
// torch::Tensor NMS(torch::Tensor kpts);

class SPDetector {
public:
    SPDetector(std::shared_ptr<SuperPoint> _model);
    void detect(cv::Mat &image);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<SuperPoint> model;
    Tensor mProb;
    Tensor mDesc;
};


// #########################  DEFINITION  ###############################

// // Superpoint Constructor
// SuperPoint::SuperPoint():
//     conv1a(Conv2dOptions( 1, c1, 3).stride(1).padding(1)),
//     conv1b(Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

//     conv2a(Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
//     conv2b(Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

//     conv3a(Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
//     conv3b(Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

//     conv4a(Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
//     conv4b(Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

//     convPa(Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
//     convPb(Conv2dOptions(c5, 65, 1).stride(1).padding(0)),
    
//     convDa(Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
//     convDb(Conv2dOptions(c5, d1, 1).stride(1).padding(0))
// { 

//     register_module("conv1a", conv1a);
//     register_module("conv1b", conv1b);

//     register_module("conv2a", conv2a);
//     register_module("conv2b", conv2b);

//     register_module("conv3a", conv3a);
//     register_module("conv3b", conv3b);

//     register_module("conv4a", conv4a);
//     register_module("conv4b", conv4b);

//     register_module("convPa", convPa);
//     register_module("convPb", convPb);

//     register_module("convDa", convDa);
//     register_module("convDb", convDb);
// }

    /*
    A distinction is made between three kinds of persistent data 
    that may be associated with a Module:

      1. Parameters: tensors that record gradients, typically weights 
        updated during the backward step (e.g. the weight of a Linear module),

      2. Buffers: tensors that do not record gradients, typically updated 
        during the forward step, such as running statistics (e.g. mean and variance in the BatchNorm module),

      3. Any additional state, not necessarily tensors, 
        required for the implementation or configuration of a Module.
    */

//  Lastly, registered parameters and buffers are handled specially during a clone() operation, 
//  which performs a deepcopy of a cloneable Module hierarchy.

std::vector<Tensor> SuperPoint::forward(Tensor input) {
    
    //SHARED ENCODER
    auto x = relu(conv1a->forward(input));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    //DETECTOR
    auto cPa = relu(convPa->forward(x));
    auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]
    
    //DESCRIPTOR
    auto cDa = relu(convDa->forward(x));
    auto desc = convDb->forward(cDa);  // [B, 256, H/8, W/8]

    if(Explain) desc.print();
    auto dn = norm(desc, 2, 1);
    if(Explain) dn.print();
    desc = desc.div(unsqueeze(dn, 1));
    if(Explain) desc.print();

    if(Explain) semi.print();
    semi = softmax(semi, 1);
    if(Explain) semi.print();
    semi = semi.slice(1, 0, 64);
    if(Explain) semi.print();
    semi = semi.permute({0, 2, 3, 1});  // [B, H/8, W/8, 64]
    if(Explain) semi.print();


    int Hc = semi.size(1);
    int Wc = semi.size(2);
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    if(Explain) semi.print();
    semi = semi.permute({0, 1, 3, 2, 4});
    if(Explain) semi.print();
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8});  // [B, H, W]
    if(Explain) semi.print();


    std::vector<Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
  }

}

#endif