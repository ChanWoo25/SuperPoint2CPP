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

    // Default Constuctor. No Use.
    SPDetector();

    /**
     * @brief Construct a new SPDetector::SPDetector object. 
     * When Initialize SPDetector, (1) First we initialize 
     * SuperPoint Class with weight_dir and use_cuda arguments. 
     * (2) and Move to device(cpu or gpu) we'll use. 
     * (3) Make the model eveluation mode, too.
     * 
     * @param _weight_dir the PATH that contains pretrained weight.
     * @param _use_cuda whether the model operates in cpu or gpu.
     */
    SPDetector(std::string _weight_dir, bool _use_cuda);
    
    /**
     * @brief Detect input image's Keypoints and Compute Descriptor.
     * 
     * @param img Input image. We use img's deep copy object.
     * @return cv::Mat 
     */
    cv::Mat *detect(cv::Mat &img);

    void fast_nms(cv::Mat& desc_no_nms, cv::Mat& desc_nms, int img_width, int img_height);
    
    void NMS2
    (std::vector<cv::KeyPoint> det, cv::Mat conf, 
     std::vector<cv::KeyPoint>& pts, int border, 
     int dist_thresh, int img_width, int img_height);
    
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    
    void computeDescriptors(cv::Mat& descriptors);

    std::vector<KeyPointNode> kpts_node_nms;
    std::vector<cv::KeyPoint> kpts_nms;
    cv::Mat kpts_loc;               ///
    cv::Mat kpts_conf;              ///
    cv::Mat descriptors;
    int n_keypoints;

private:
    std::shared_ptr<SuperPoint> model;  /// Superpoint model                
    

    // kFloat32, kStrided, requires_grad(false), cpu or gpu device.
    c10::TensorOptions tensor_opts;     
    c10::DeviceType mDeviceType;      
    c10::Device mDevice;

    // Superpoint Output Probability Tensor
    torch::Tensor mProb;              
    // Superpoint Output Descriptor Tensor
    torch::Tensor mDesc;              
    
    void SemiNMS(at::Tensor& kpts);
                    
    float conf_thres=0.0625;             /// 각 픽셀의 기댓값: 1/64 = 0.015625
    float nn_thres;                     ///
    bool verbose = 0;                   ///
    
    // NMS parameters. (Not use now.)
    bool nms = false;
    int MAX_KEYPOINT = 100;           
    int nms_border = 8;               
    int nms_dist_thres = 4;           
    float nms_dist;   
};

}


#endif