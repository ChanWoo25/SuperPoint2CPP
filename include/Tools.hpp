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
    cv::Size input_size = {320, 240}; // {Width, Height}
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
    VideoStreamer(const cv::String& filename){
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
    cv::Mat read_image(const cv::String& path);
    // Read a image as grayscale and resize to img_size.

    bool next_frame();
    void setImageSize(const cv::Size &sz){ input_size = sz; 
        W_scale = (float)image_size.width / (float)input_size.width;
        H_scale = (float)image_size.height / (float)input_size.height;
    }


};


#endif