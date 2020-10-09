#include <Tools.hpp>

cv::Mat VideoStreamer::read_image(const cv::String& path)
{
    cv::Mat gray_img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    // resampling using pixel area relation
    // resize(input, output, Size, scale_factor_x, scale_factor_y, interpolation_method)
    if(pImgSize != NULL)
        cv::resize(gray_img, gray_img, (*pImgSize), 0, 0, cv::INTER_AREA);
    
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

        if(pImgSize != NULL)
            cv::resize(input, input, (*pImgSize), 1., 1., cv::INTER_AREA);
        
        cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
        input.convertTo(input, CV_32F);
    }
    else if(img_source == input_device::IS_VIDEO_FILE)
    {
        /* Read next image. Return false if failed.*/
        if(!cap.read(img))
        {
            std::cout << "No Image.\n";
            return false;
        }
        
        input = img.clone();
        cv::cvtColor(input, input, cv::COLOR_RGB2GRAY);
        
        /* If the size is fixed, adjust it. */
        if(pImgSize != NULL)
        {
            W_scale = (float)img.size().width / (float)pImgSize->width;
            H_scale = (float)img.size().height / (float)pImgSize->height;
            cv::resize(input, input, (*pImgSize), 1., 1., cv::INTER_AREA);
        }

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

std::string cv_type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}