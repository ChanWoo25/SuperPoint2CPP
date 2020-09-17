#ifndef SUPERPOINT_HPP
#define SUPERPOINT_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>

namespace SuperPointSLAM
{

using namespace torch;
using namespace nn;

/**
 * @brief 순수 SuperPoint 클래스
 */
class SuperPoint : public Module {
public:
    // SuperPoint Model Constructor
    SuperPoint();

    /**
     * Display some information
     * 1. Cuda Availability 
     * 2. GPU number 
     * 3. cudnn availability.
     */
    void display();

    // Display Module and Submodule's detail informations.
    // (1)Whether it is trainable 
    // (2)module's name(ex. Conv2D or Linear etc.).
    void display(std::shared_ptr<SuperPoint> net);

    /**
     * @brief Forward propagation
     * @param x input Tensor.
     * @param Prob Output Probabities Tensor
     * @param Desc Output Descriptors Tensor
     * @details Return probabilities and descriptors using Prob, Desc.
     * - all Arguments are by reference to speed up.
     * - When input x's demension is [B, H, W], (B = Batch_size)
     * - Output dimension is as follows.
     * - Prob: [B, H, W]
     * - Desc: [B, 256, H/8, W/8]
     */
    void forward(torch::Tensor& x, torch::Tensor& Prob, torch::Tensor& Desc);

protected:
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

    const float EPSILON = 1e-19;
};

}

#endif