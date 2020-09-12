#ifndef SPDETECTOR_HPP
#define SPDETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <iostream>
#include <SuperPoint.hpp>

namespace SuperPointSLAM
{

class SPDetector
{
private:
    struct KeyPointNode{
        cv::KeyPoint kpt;
        int desc_idx;
    };

public:
    SPDetector();
    SPDetector(std::string _weight_dir, bool _use_cuda);
    

    void fast_nms(cv::Mat& desc_no_nms, cv::Mat& desc_nms, int img_width, int img_height);
     
    cv::Mat detect(cv::Mat &img);
    
    void NMS2
    (std::vector<cv::KeyPoint> det, cv::Mat conf, 
     std::vector<cv::KeyPoint>& pts, int border, 
     int dist_thresh, int img_width, int img_height);
    
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    
    void computeDescriptors(cv::Mat& descriptors);

    std::vector<KeyPointNode> kpts_node_nms;
    std::vector<cv::KeyPoint> kpts_nms;

private:
    std::shared_ptr<SuperPoint> model;  /// Superpoint model                
    cv::Mat kpts_nms_loc;               ///
    cv::Mat kpts_nms_conf;              ///
    c10::TensorOptions tensor_opts;     ///
    c10::DeviceType device_type;        ///
    torch::Tensor mProb;                ///
    torch::Tensor mDesc;                ///
    int MAX_KEYPOINT = 100;            ///
    int nms_border = 6;                 ///
    int nms_dist_thres = 3;             ///
    bool use_cuda;                      ///
    float nms_dist;                     ///
    float conf_thres=0.002;             ///
    float nn_thres;                     ///
    bool verbose = 0;                   ///
};

}


#endif