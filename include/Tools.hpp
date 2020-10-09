#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>

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
    cv::Size *pImgSize = NULL;
    
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
    VideoStreamer(int cam_id):img_source(input_device::IS_CAMERA)
    {
        deviceID = cam_id;
        cap.open(deviceID, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }

        cv::Mat test_grab;
        while(!cap.read(test_grab));
        pImgSize = new cv::Size(test_grab.size());
    }
    VideoStreamer(const cv::String& filename):img_source(input_device::IS_VIDEO_FILE)
    {
        cap.open(filename, apiID);
        if (!cap.isOpened()) {
            std::cerr << "ERROR! Unable to open camera\n";
            std::exit(1);
        }

        // cv::Mat test_grab;
        // while(!cap.read(test_grab));
        // image_size = test_grab.size();
        // W_scale = (float)image_size.width / (float)input_size.width;
        // H_scale = (float)image_size.height / (float)input_size.height;
    }
    
    float H_scale=1.0, W_scale=1.0;
    cv::Mat img, input;
    cv::Mat read_image(const cv::String& path);
    // Read a image as grayscale and resize to img_size.

    bool next_frame();
    void setImageSize(const cv::Size &_size){ 
        pImgSize = new cv::Size(_size); 
    }
};

std::string cv_type2str(int type);

#endif