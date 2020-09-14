#include <Tools.hpp>

cv::Mat VideoStreamer::read_image(const cv::String& path)
{
    cv::Mat gray_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    // resampling using pixel area relation
    // resize(input, output, Size, scale_factor_x, scale_factor_y, interpolation_method)
    cv::resize(gray_img, gray_img, input_size, 0, 0, cv::INTER_AREA);
    if(gray_img.empty()){
        std::cerr << "Error reading image.\n";
        exit('2');
    }
    gray_img.convertTo(gray_img, CV_32F);
    return gray_img/255.;
}

bool VideoStreamer::next_frame()
{
    if(current_frame_num >= MAX_FRAME_NUM) return false;
    if(img_source == input_device::IS_CAMERA)
    {
        if(!cap.read(img))
        {
            std::cout << "No Image.\n";
            return false;
        }
        input = img.clone();
        cv::resize(input, input, input_size, 1., 1., cv::INTER_AREA);
        cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
        input.convertTo(input, CV_32F);
    }
    else if(img_source == input_device::IS_VIDEO_FILE)
    {
        if(!cap.read(img))
        {
            std::cout << "No Image.\n";
            return false;
        }
        
        input = img.clone();
        cv::resize(input, input, input_size, 1., 1., cv::INTER_AREA);
        cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
        input.convertTo(input, CV_32F);
    }
    else
    {
        std::cerr << "There is no source of image frames!\n";
        exit(2);
    }
    current_frame_num++;
    return true;
}
