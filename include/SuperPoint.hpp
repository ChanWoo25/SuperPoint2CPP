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

void printSection(int n, std::string s);
std::string cv_type2str(int type);
void test_with_magicleap();
cv::Mat argmin_cv_mat(const cv::Mat& mat, int axis);
cv::Mat argmin_cv_mat_with_score(const cv::Mat& mat, int axis, cv::Mat& score);

/**
 * @brief 순수 SuperPoint 클래스
 */
class SuperPoint : public Module {
public:
    //Constructor
    SuperPoint();

    /**
     * @brief Display some information - 1. Cuda Availability 
     * 2. GPU number 3. cudnn availability.
     */
    void display();

    // Display Module and Submodule's detail informations.
    // (1)Whether it is trainable 
    // (2)module's name(ex. Conv2D or Linear etc.).
    void display(std::shared_ptr<SuperPoint> net);

    // Forward propagation
    void forward(torch::Tensor x, torch::Tensor& Prob, torch::Tensor& Desc);

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
    bool verbose = 0;
};


}

#endif



// cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms);
// // torch::Tensor NMS(torch::Tensor kpts);

// class SPDetector {
// public:
//     SPDetector(std::shared_ptr<SuperPoint> _model);
//     void detect(cv::Mat &image);
//     void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
//     void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

// private:
//     std::shared_ptr<SuperPoint> model;
//     Tensor mProb;
//     Tensor mDesc;
// };
