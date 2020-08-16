#ifndef SUPERPOINT_H
#define SUPERPOINT_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <vector>

namespace SUPERPOINT
{
using namespace torch;
using namespace nn;

class SuperPoint : public Module {
// class를 쓰면, 권한을 지정해야한다. 상속시 public Module을 바꿨고,
// 멤버함수, 변수들도 public, private지정을 안해주면 모두 private처리가 되기 때문에 forward에 접할 수 없게된다.
public:
  SuperPoint(); //생성자

  std::vector<torch::Tensor> forward(Tensor input); //순전파

private:
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

}  // ORB_SLAM


#endif