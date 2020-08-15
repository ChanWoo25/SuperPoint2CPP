#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <vector>

#ifdef EIGEN_MPL2_ONLY
#undef EIGEN_MPL2_ONLY
#endif


namespace SUPERPOINT
{
using namespace torch;
using namespace nn;

struct SuperPoint : Module {
  SuperPoint(); //생성자

  std::vector<torch::Tensor> forward(Tensor x);

  //SHARED ENCODER
  Conv2d conv1a;
  Conv2d conv1b;

  Conv2d conv2a;
  Conv2d conv2b;

  Conv2d conv3a;
  Conv2d conv3b;

  Conv2d conv4a;
  Conv2d conv4b;

  //DETECTOR
  Conv2d convPa;
  Conv2d convPb;

  //DESCRIPTOR
  Conv2d convDa;
  Conv2d convDb;

  const int c1 = 64;
  const int c2 = 64;
  const int c3 = 128;
  const int c4 = 128;
  const int c5 = 256;
  const int d1 = 256;
};


cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms, bool cuda);
// torch::Tensor NMS(torch::Tensor kpts);

class SPDetector {
public:
    SPDetector(std::shared_ptr<SuperPoint> _model);
    void detect(cv::Mat &image, bool cuda);
    void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
    void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

private:
    std::shared_ptr<SuperPoint> model;
    Tensor mProb;
    Tensor mDesc;
};

}  // ORB_SLAM


#endif