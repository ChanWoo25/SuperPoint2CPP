/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/





// cv::Mat SPdetect(std::shared_ptr<SuperPoint> model, cv::Mat img, std::vector<cv::KeyPoint> &keypoints, double threshold, bool nms);
// // torch::Tensor NMS(torch::Tensor kpts);

// class SPDetector {
// public:
//     SPDetector(std::shared_ptr<SuperPoint> _model);
//     void detect(cv::Mat &image);
//     void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms);
//     void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

// private:
//     std::shared_ptr<SuperPoint> model;
//     Tensor mProb;
//     Tensor mDesc;
// };



#include <SPDetector.hpp>

namespace SuperPointSLAM
{

SPDetector::SPDetector(std::string _weight_dir, bool _use_cuda)
    :   mDeviceType((_use_cuda) ? c10::kCUDA : c10::kCPU),
        mDevice(c10::Device(mDeviceType))
{   
    /** CONSTRUCTOR **/

    model = std::make_shared<SuperPoint>();
    torch::load(model, _weight_dir);

    // mDeviceType = (_use_cuda) ? c10::kCUDA : c10::kCPU;
    // mDevice = c10::Device(mDeviceType);

    // This options aren't allow to be changed.
    tensor_opts = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(c10::kStrided)
                        .requires_grad(false);
    if (_use_cuda)
        model->to(mDevice);
    model->eval();
}

cv::Mat SPDetector::detect(cv::Mat &img)
{
    at::Tensor x = torch::from_blob((void*)img.clone().data, \
                                    {1, 1, img.rows, img.cols}, \
                                    tensor_opts).to(mDevice);
    x /= 255.0;

    torch::Tensor prob, desc;
    model->forward(x, prob, desc);
    prob = prob.squeeze(0);
    mProb = prob.clone();// [H, W] [120, 160]
    mDesc = desc.clone();// [1, 256, H/8, W/8]

    at::Tensor kpts = (prob > conf_thres);          
    kpts = at::nonzero(kpts); // [N, 2] (y, x)               
    at::Tensor fkpts = kpts.to(kFloat);
    at::Tensor grid = torch::zeros({1, 1, kpts.size(0), 2}).to(mDevice);


    /** Get each Keypoints' descriptor. **/ 
    grid[0][0].slice(1, 0, 1) = (2.0 * (fkpts.slice(1, 1, 2) / prob.size(1))) - 1; // x
    grid[0][0].slice(1, 1, 2) = (2.0 * (fkpts.slice(1, 0, 1) / prob.size(0))) - 1; // y
    desc = at::grid_sampler(desc, grid, 0, 0, true);    // [1, 256, 1, n_keypoints]
    desc = desc.squeeze(0).squeeze(1);                  // [256, n_keypoints]

    /** Normalize 1-Dimension with 2-Norm. **/
    at::Tensor dn = at::norm(desc, 2, 1);       // [256, 1]
    desc = desc.div(unsqueeze(dn, 1));          // [256, n_keypoints]
    desc = desc.transpose(0, 1).contiguous();   // [n_keypoints, 256]

    // After processing, back to CPU only descriptor
    if (mDeviceType == c10::kCUDA)
        desc = desc.to(kCPU);

    //  Convert descriptor 
    //  From at::Tensor To cv::Mat
    cv::Size desc_size(desc.size(1), desc.size(0)); // [256, n_keypoints]
    int n_keypoints = desc.size(0); 
    std::cout << "Before nms: " << n_keypoints << std::endl;
    cv::Mat desc_no_nms(desc_size, CV_32FC1, desc.data_ptr<float>());
    //cv::Mat desc_no_nms(desc_size, CV_32F);
    //std::memcpy(desc.data_ptr(), desc_no_nms.data, sizeof(float) * desc.numel());

    // Convert Keypoint
    // From torch::Tensor   kpts(=keypoints)
    // To   cv::KeyPoint    keypoints_no_nms
    kpts_node_nms.clear();
    for (int i = 0; i < n_keypoints; i++)
    {
        float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
        cv::KeyPoint kpt(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 4, -1, response);
        kpts_node_nms.push_back({kpt, i});
    }

    cv::Mat desc_nms;
    fast_nms(desc_no_nms, desc_nms, img.cols, img.rows);
    std::cout << "After nms: " << kpts_nms.size() << std::endl;
    
    /*지금은 필요없음
        // Empty cv::Mat that will update.
        kpts_nms_loc.create(kpts_node_nms.size(), 2, CV_32F); 
        kpts_nms_conf.create(kpts_node_nms.size(), 1, CV_32F); 
        auto xy         = kpts_nms_loc.ptr<float>(0);
        auto conf_ptr   = kpts_nms_conf.ptr<float>(0);
        std::cout << (d_i++) << std::endl;

        for(auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
        {
            *(xy++) = (float)(*iter).kpt.pt.x;
            *(xy++) = (float)(*iter).kpt.pt.y;
            *(conf_ptr++)   = (float)(*iter).kpt.response; 
        }
        std::cout << (d_i++) << std::endl;
    */
    mProb.reset();
    mDesc.reset();
    return desc_nms;
}

void SPDetector::fast_nms(cv::Mat& desc_no_nms, cv::Mat& desc_nms, int img_width, int img_height)
{
    //std::cout << "desc_no_nms Type :" << desc_no_nms.size() << std::endl;

    // kpys_no_nms: The keypoints' vectorsorted by conf value
    // Empty cv::Mat that will update.
    auto ptr = *kpts_node_nms.begin();

    // Sorting keypoints by reference value.
    std::sort(kpts_node_nms.begin(), kpts_node_nms.end(), 
            [](KeyPointNode a, KeyPointNode b) -> bool{ return a.kpt.response > b.kpt.response; });
    
    // std::cout << "<response order>" << std::endl;
    // for(int i=0; i<5;i++)
    //     std::cout << kpts_node_nms[i].kpt.response << std::endl;

    // cv::Mat kpt_mat(kpts_node_nms.size(), 2, CV_32F);    //  [n_keypoints, 2]
    // cv::Mat conf(kpts_node_nms.size(), 1, CV_32F);       //  [n_keypoints, 1]
    
    // auto xy         = kpt_mat.ptr<float>(0);
    // auto conf_ptr   = conf.ptr<float>(0);

    // for(auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
    // {
    //     *(xy++) = (float)(*iter).kpt.pt.x;
    //     *(xy++) = (float)(*iter).kpt.pt.y;
    //     *(conf_ptr++)   = (float)(*iter).kpt.response; 
    // }
    // std::cout << (d_i++) << std::endl;


    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8U, cv::Scalar(0));
    //cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16U, cv::Scalar(0));
    //cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32F, cv::Scalar(0));

    int nms_idx(0);
    for (auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
    {
        int col = (int)(*iter).kpt.pt.x;
        int row = (int)(*iter).kpt.pt.y;

        grid.at<char>(row, col) = (char)1;
        //inds.at<unsigned short>(vv, uu) = (nms_idx++);
        //confidence.at<float>(vv, uu) = (*iter).kpt.response;
    }

    // Padding grid Mat.
    //  cv::copyMakeBorder(intputArr, outputArr, offset * 4, BorderType, BorderScalar)
    //      input, output arr를 따로 지정할 수 있으나 여기서는 in-place 수행을 위해 grid, grid
    //      사방면에 border 길이를 지정해주고, How는 bordertype을 통해 지정.
    //      constant로 채운다고 했으므로 어떤 값으로 채울지 BorderScalar(0)로 넘김.
    int d(nms_dist_thres), b(nms_border);
    cv::copyMakeBorder(grid, grid, d, d, d, d, cv::BORDER_CONSTANT, 0);

    // Process Non-Maximum Suppression from highest confidence Keypoint.
    // find Keypoints in range of nms_dist_thres and set 0.

    // 하나의 for문으로 해결하기 위해 노력했다.
    // 허락된 Boundary 안쪽의 높은 Confidence를 가진 Keypoint부터 시작하여
    // 자기 주변 distance 안의 자신보다 낮은 Confidence를 지닌 Keypoint를 제거한다.
    // Input은 함수 인자로 받지만, output은 SuprpointFrontend의 멤버 변수로 저장한다.
    int cnt = 0;
    kpts_nms.clear();
    for (auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
    {
        int col = (int)(*iter).kpt.pt.x + d;
        int row = (int)(*iter).kpt.pt.y + d;
        if(col <= b | row <= b | col >= (img_width - d) | row >= (img_height - d))
            continue;
        
        auto center = grid.ptr<char>(row) + col;
        //auto center_conf = confidence.ptr<float>(v) + u;
        if (*center == 1)
        {
            cv::Mat sub(grid, cv::Rect(cv::Point(col-d, row-d), cv::Point(col+d, row+d)));
            sub.setTo(0);
            cnt++;

            // If extract 300 keypoints, it's enough, Break.
            if(cnt >= MAX_KEYPOINT) break;
            kpts_nms.push_back((*iter).kpt);

            //desc_nms.push_back(desc_no_nms.row((*iter).desc_idx));
            auto dec = desc_no_nms.row((*iter).desc_idx);
            desc_nms.push_back(dec);
            //std::cout << desc_nms.size() << std::endl;
        }
        else{ continue; }
    }
}

}//SuperPointSLAM