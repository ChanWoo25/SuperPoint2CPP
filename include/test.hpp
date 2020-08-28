#ifndef TEST_H
#define TEST_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>

namespace NAMU_TEST
{

using namespace torch;
using namespace nn;

void printSection(int n, std::string s);
std::string cv_type2str(int type);
void test_with_magicleap();
cv::Mat argmin_cv_mat(const cv::Mat& mat, int axis);
cv::Mat argmin_cv_mat_with_score(const cv::Mat& mat, int axis, cv::Mat& score);
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

class SuperPointFrontend
{
private:
    struct KeyPointNode{
        cv::KeyPoint kpt;
        int desc_idx;
    };

public:
    SuperPointFrontend();
    SuperPointFrontend(std::string _weight_dir, bool _use_cuda);
    

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
    std::shared_ptr<SuperPoint> model;
    cv::Mat kpts_nms_loc;
    cv::Mat kpts_nms_conf;
    c10::TensorOptions tensor_opts;
    c10::DeviceType device_type;
    torch::Tensor mProb;
    torch::Tensor mDesc;
    int MAX_KEYPOINT = 100;
    int nms_border = 8;
    int nms_dist_thres = 4;
    bool use_cuda;
    float nms_dist;
    float conf_thres=0.1;
    float nn_thres;
    bool verbose = 0;
};

class VideoStreamer{
private:
    enum class input_device{
        IS_CAMERA=0, 
        IS_VIDEO_FILE=1
    };
    input_device img_source;
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    int MAX_FRAME_NUM = 1000000;
    int current_frame_num = 0;
    cv::Size input_size = {160, 120}; // {Width, Height}
    cv::Size image_size;
    
public:
    VideoStreamer(){
        cap.open(deviceID, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }
        img_source = input_device::IS_CAMERA;
    }
    ~VideoStreamer()
    {
        // When everything done, release the video capture object
        cap.release();
    }
    VideoStreamer(int cam_id)
    {
        deviceID = cam_id;
        cap.open(deviceID, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }
        img_source = input_device::IS_CAMERA;
        cv::Mat test_grab;
        while(!cap.read(test_grab));
        image_size = test_grab.size();
        W_scale = (float)image_size.width / (float)input_size.width;
        H_scale = (float)image_size.height / (float)input_size.height;
        
    }
    VideoStreamer(const string& filename){
        cap.open(filename, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }
        img_source = input_device::IS_VIDEO_FILE;
        cv::Mat test_grab;
        while(!cap.read(test_grab));
        image_size = test_grab.size();
        W_scale = (float)image_size.width / (float)input_size.width;
        H_scale = (float)image_size.height / (float)input_size.height;
    }
    float H_scale, W_scale;
    cv::Mat img, input;
    cv::Mat read_image(const string& path);
    // Read a image as grayscale and resize to img_size.

    bool next_frame();


};

class Tracker{
/*
    Class to manage a fixed memory of points and descriptors 
    that enables sparse optical flow point tracking.

    Internally, the tracker stores a 'tracks' matrix 
    sized M x (2+L), of M tracks with maximum length L, 
    where each row corresponds to:
        row_m = [track_id_m, avg_desc_score_m, point_id_0_m, 
                ..., point_id_L-1_m].
*/
private:
    int MAX_LEN;
    int MAX_SCORE=9999;
    float nn_thres;
    int track_cnt=0;
    cv::Mat last_desc;
    std::vector<cv::Point> all_pts;
    
public:
    Tracker();
    cv::Mat nn_match_two_way(const cv::Mat& desc1, const cv::Mat& desc2);
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

