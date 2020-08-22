#ifndef TEST_H
#define TEST_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>

namespace NAMU_TEST
{

using namespace torch;
using namespace nn;

void printSection(int n, std::string s);


class SuperPoint : public Module {
public:
    //Constructor
    SuperPoint();

    // Display some information.
    // 1. Cuda Availability. && 2. GPU number.
    // 3. cudnn availability.
    void display();

    // Display Module and Submodule's detail informations.
    // (1)Whether it is trainable 
    // (2)module's name(ex. Conv2D or Linear etc.).
    void display(std::shared_ptr<SuperPoint> net);

    // Forward propagation
    std::vector<torch::Tensor> forward(torch::Tensor x);

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

class SuperPointFrontend
{
public:
    SuperPointFrontend();
    
    SuperPointFrontend(std::string _weight_dir, bool _use_cuda);
    
    cv::Mat detect(cv::Mat &img);
    
    void NMS
    (const cv::Mat& kpts_loc, const cv::Mat& kpts_conf, const cv::Mat& desc_no_nms, 
     std::vector<cv::KeyPoint> &kpt_nms, cv::Mat &desc_nms, 
     int border, int dist_thresh, int img_width, int img_height);
    
    void fast_nms
    (const std::vector<cv::KeyPoint>& kypts_no_nms, const cv::Mat& desc_no_nms,
     int border, int dist_thresh, int img_width, int img_height);

    void NMS2
    (std::vector<cv::KeyPoint> det, cv::Mat conf, 
     std::vector<cv::KeyPoint>& pts, int border, 
     int dist_thresh, int img_width, int img_height);
    
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<SuperPoint> model;
    c10::TensorOptions tensor_opts;
    c10::DeviceType device_type;
    torch::Tensor mProb;
    torch::Tensor mDesc;
    int nms_border = 8;
    int nms_dist_thres = 4;
    bool use_cuda;
    float nms_dist;
    float conf_thres;
    float nn_thres;
    int output_cell = 8;
    int range_border = 4;
    bool verbose = 0;
};

cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms);
// torch::Tensor NMS(torch::Tensor kpts);

class SPDetector {
public:
    SPDetector(std::shared_ptr<SuperPoint> _model);
    void detect(cv::Mat &img);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<SuperPoint> model;
    torch::Tensor mProb;
    torch::Tensor mDesc;
};

}// namespace NAMU_TEST END



#endif

