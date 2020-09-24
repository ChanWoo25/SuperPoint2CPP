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

cv::Mat* SPDetector::detect(cv::Mat &img)
{
    bool scale = true;

    at::Tensor x = torch::from_blob((void*)img.clone().data, \
                                    {1, 1, img.rows, img.cols}, \
                                    tensor_opts).to(mDevice);
    x = (x + EPSILON) / 255.0;

    model->forward(x, mProb, mDesc);
    mProb = mProb.squeeze(0);

    //CUDA bool type을 반환, GPU에 있는 원소는 접근 불가능.
    at::Tensor kpts = (mProb > conf_thres);  
    
    // 중복된 위치 제거.
    SemiNMS(kpts);


    kpts = at::nonzero(kpts); // [N, 2] (y, x)               
    at::Tensor fkpts = kpts.to(kFloat);
    at::Tensor grid = torch::zeros({1, 1, kpts.size(0), 2}).to(mDevice); 
    // grid.print(); // [CUDAFloatType [1, 1, 225, 2]]

    // mProb size(1): W - cols - 320, size(0): H - rows - 240

    /** Get each Keypoints' descriptor. **/ 
    grid[0][0].slice(1, 0, 1) = (2.0 * (fkpts.slice(1, 1, 2) / mProb.size(1))) - 1; // x
    grid[0][0].slice(1, 1, 2) = (2.0 * (fkpts.slice(1, 0, 1) / mProb.size(0))) - 1; // y
    mDesc = at::grid_sampler(mDesc, grid, 0, 0, false);    // [1, 256, 1, n_keypoints]
    mDesc = mDesc.squeeze(0).squeeze(1);                  // [256, n_keypoints]

    /** Normalize 1-Dimension with 2-Norm. **/
    at::Tensor dn = at::norm(mDesc, 2, 1);          // [CUDAFloatType [256]]
    mDesc = at::div((mDesc + EPSILON), unsqueeze(dn, 1));
    //mDesc = mDesc.div(unsqueeze(dn, 1));            // [256, n_keypoints] <- unsqueeezed dn[CUDAFloatType [256, 1]]
    mDesc = mDesc.transpose(0, 1).contiguous();     // [CUDAFloatType [N, 256]]
    
    // After processing, back to CPU only descriptor
    if (mDeviceType == c10::kCUDA)
        mDesc = mDesc.to(kCPU);

    /** Convert descriptor From at::Tensor To cv::Mat **/  
    cv::Size desc_size(mDesc.size(1), mDesc.size(0)); 
    n_keypoints = mDesc.size(0); 
    std::cout << "N - Keypoint : " << n_keypoints << std::endl;
    
    // std::cout << "mDesc.numel() : " << mDesc.numel() << std::endl;
    // std::cout << "mDesc.numel() * float : " << sizeof(float) * mDesc.numel() << std::endl;
    // [256, N], CV_32F
    descriptors.create(n_keypoints, 256, CV_32FC1);

    memcpy((void*)descriptors.data, mDesc.data_ptr(), sizeof(float) * mDesc.numel());
    // descriptors = cv::Mat(desc_size, CV_32FC1, mDesc.data_ptr<float>());
    

    // Convert Keypoint
    // From torch::Tensor   kpts(=keypoints)
    // To   cv::KeyPoint    keypoints_no_nms
    kpts_loc.create(fkpts.size(1), fkpts.size(0), CV_32FC1);    // [N, 2] (x, y)     
    kpts_conf.create(1, fkpts.size(0), CV_32FC1);               // [N, 1]
    
    auto ploc = kpts_loc.ptr<float>();
    auto pconf = kpts_conf.ptr<float>();
    for (int i = 0; i < n_keypoints; i++)
    {
        *(ploc++) = kpts[i][1].item<float>();
        *(ploc++) = kpts[i][0].item<float>();
        *(pconf++) = mProb[kpts[i][0]][kpts[i][1]].item<float>();
    }
    
    mProb.reset();
    mDesc.reset();

    return &descriptors;
}

void SPDetector::SemiNMS(at::Tensor& kpts)
{
    if (mDeviceType == c10::kCUDA)
        kpts = kpts.to(kCPU);
    // std::cout << kpts.scalar_type() << sizeof(kpts.scalar_type()) << std::endl;
    // NMS alternative
    int rowlen = kpts.size(0);
    int collen = kpts.size(1);

    //auto accessor = kpts.accessor<bool,2>();
    auto pT1 = kpts.data_ptr<bool>();
    auto pT2 = pT1 + collen;
    auto pT3 = pT2 + collen;

    for(int i = 0; i < rowlen; i++)
    {
        for(int j = 0 ; j < collen; j++)
        {
            if(*pT1 && (i < rowlen-2) && (j < collen-2))
            {
                *(pT1 + 1) = 0; *(pT1 + 2) = 0;
                *pT2 = 0; *(pT2 + 1) = 0; *(pT2 + 2) = 0; 
                *pT3 = 0; *(pT3 + 1) = 0; *(pT3 + 2) = 0; 
            }
            pT1++;
            pT2++;
            pT3++;
        }
    }

    if (mDeviceType == c10::kCUDA)
        kpts = kpts.to(kCUDA);
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

// void SPDetector::new_fast_nms(at::Tensor *desc_no_nms, at::Tensor *desc_nms, int img_width, int img_height)
// {
//     //std::cout << "desc_no_nms Type :" << desc_no_nms.size() << std::endl;

//     // kpys_no_nms: The keypoints' vectorsorted by conf value
//     // Empty cv::Mat that will update.
//     auto ptr = *kpts_node_nms.begin();

//     // Sorting keypoints by reference value.
//     std::sort(kpts_node_nms.begin(), kpts_node_nms.end(), 
//             [](KeyPointNode a, KeyPointNode b) -> bool{ return a.kpt.response > b.kpt.response; });
    
//     // std::cout << "<response order>" << std::endl;
//     // for(int i=0; i<5;i++)
//     //     std::cout << kpts_node_nms[i].kpt.response << std::endl;

//     // cv::Mat kpt_mat(kpts_node_nms.size(), 2, CV_32F);    //  [n_keypoints, 2]
//     // cv::Mat conf(kpts_node_nms.size(), 1, CV_32F);       //  [n_keypoints, 1]
    
//     // auto xy         = kpt_mat.ptr<float>(0);
//     // auto conf_ptr   = conf.ptr<float>(0);

//     // for(auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
//     // {
//     //     *(xy++) = (float)(*iter).kpt.pt.x;
//     //     *(xy++) = (float)(*iter).kpt.pt.y;
//     //     *(conf_ptr++)   = (float)(*iter).kpt.response; 
//     // }
//     // std::cout << (d_i++) << std::endl;


//     cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8U, cv::Scalar(0));
//     //cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16U, cv::Scalar(0));
//     //cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32F, cv::Scalar(0));

//     int nms_idx(0);
//     for (auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
//     {
//         int col = (int)(*iter).kpt.pt.x;
//         int row = (int)(*iter).kpt.pt.y;

//         grid.at<char>(row, col) = (char)1;
//         //inds.at<unsigned short>(vv, uu) = (nms_idx++);
//         //confidence.at<float>(vv, uu) = (*iter).kpt.response;
//     }

//     // Padding grid Mat.
//     //  cv::copyMakeBorder(intputArr, outputArr, offset * 4, BorderType, BorderScalar)
//     //      input, output arr를 따로 지정할 수 있으나 여기서는 in-place 수행을 위해 grid, grid
//     //      사방면에 border 길이를 지정해주고, How는 bordertype을 통해 지정.
//     //      constant로 채운다고 했으므로 어떤 값으로 채울지 BorderScalar(0)로 넘김.
//     int d(nms_dist_thres), b(nms_border);
//     cv::copyMakeBorder(grid, grid, d, d, d, d, cv::BORDER_CONSTANT, 0);

//     // Process Non-Maximum Suppression from highest confidence Keypoint.
//     // find Keypoints in range of nms_dist_thres and set 0.

//     // 하나의 for문으로 해결하기 위해 노력했다.
//     // 허락된 Boundary 안쪽의 높은 Confidence를 가진 Keypoint부터 시작하여
//     // 자기 주변 distance 안의 자신보다 낮은 Confidence를 지닌 Keypoint를 제거한다.
//     // Input은 함수 인자로 받지만, output은 SuprpointFrontend의 멤버 변수로 저장한다.
//     int cnt = 0;
//     kpts_nms.clear();
//     for (auto iter = kpts_node_nms.begin(); iter != kpts_node_nms.end(); iter++)
//     {
//         int col = (int)(*iter).kpt.pt.x + d;
//         int row = (int)(*iter).kpt.pt.y + d;
//         if(col <= b | row <= b | col >= (img_width - d) | row >= (img_height - d))
//             continue;
        
//         auto center = grid.ptr<char>(row) + col;
//         //auto center_conf = confidence.ptr<float>(v) + u;
//         if (*center == 1)
//         {
//             cv::Mat sub(grid, cv::Rect(cv::Point(col-d, row-d), cv::Point(col+d, row+d)));
//             sub.setTo(0);
//             cnt++;

//             // If extract 300 keypoints, it's enough, Break.
//             if(cnt >= MAX_KEYPOINT) break;
//             kpts_nms.push_back((*iter).kpt);

//             //desc_nms.push_back(desc_no_nms.row((*iter).desc_idx));
//             auto dec = desc_no_nms.row((*iter).desc_idx);
//             desc_nms.push_back(dec);
//             //std::cout << desc_nms.size() << std::endl;
//         }
//         else{ continue; }
//     }
// }

}//SuperPointSLAM

/* 비운의 코드 1 


    if(scale)
    {
        cv::Mat img2;
        cv::resize(img, img2, cv::Size(img.rows/2, img.cols/2), cv::INTER_LINEAR);
        at::Tensor x2 = torch::from_blob((void*)img2.clone().data, \
                                    {1, 1, img2.rows, img2.cols}, \
                                    tensor_opts).to(mDevice);
        x2 /= 255.0; //x2.print();
    }

*/