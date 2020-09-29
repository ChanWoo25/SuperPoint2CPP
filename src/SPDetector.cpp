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

#include <SPDetector.hpp>

namespace SuperPointSLAM
{

SPDetector::SPDetector(std::string _weight_dir, bool _use_cuda)
    :   mDeviceType((_use_cuda) ? c10::kCUDA : c10::kCPU),
        mDevice(c10::Device(mDeviceType))
{   
    /* SuperPoint model loading */
    model = std::make_shared<SuperPoint>();
    torch::load(model, _weight_dir);
    
    /* This option should be done exactly as below */
    tensor_opts = c10::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(c10::kStrided)
                        .requires_grad(false);
    if (_use_cuda)
        model->to(mDevice);
    model->eval();
}

void SPDetector::detect(cv::InputArray _image, std::vector<cv::KeyPoint>& _keypoints,
                      cv::Mat &_descriptors)
{
    cv::Mat img = _image.getMat();
    at::Tensor x = torch::from_blob((void*)img.clone().data, \
                                    {1, 1, img.rows, img.cols}, \
                                    tensor_opts).to(mDevice);
    
    // To avoid Error caused by division by zero.
    // "EPSILON" is mostly used for this purpose.
    x = (x + EPSILON) / 255.0; 

    model->forward(x, mProb, mDesc);
    mProb = mProb.squeeze(0);

    /* Return a "CUDA bool type Tensor"
     * 1 if there is a featrue, and 0 otherwise */ 
    at::Tensor kpts = (mProb > conf_thres);  
    
    /* Remove potential redundent features. */
    if(nms) 
    {   // Default=true
        SemiNMS(kpts);
    }

    /* Prepare grid_sampler() */ 
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
    //mDesc = mDesc.div(unsqueeze(dn, 1));          // [256, n_keypoints] <- unsqueeezed dn[CUDAFloatType [256, 1]]
    mDesc = mDesc.transpose(0, 1).contiguous();     // [CUDAFloatType [N, 256]]
    
    // After processing, back to CPU only descriptor
    if (mDeviceType == c10::kCUDA)
        mDesc = mDesc.to(kCPU);

    /** Convert descriptor From at::Tensor To cv::Mat **/  
    cv::Size desc_size(mDesc.size(1), mDesc.size(0)); 
    n_keypoints = mDesc.size(0); 
    
    // [256, N], CV_32F
    _descriptors.create(n_keypoints, 256, CV_32FC1);
    memcpy((void*)_descriptors.data, mDesc.data_ptr(), sizeof(float) * mDesc.numel());
    // descriptors = cv::Mat(desc_size, CV_32FC1, mDesc.data_ptr<float>());
    // std::cout << _descriptors.row(0) << std::endl;

    /* Convert Keypoint
     * From torch::Tensor   kpts(=keypoints)
     * To   cv::KeyPoint    keypoints_no_nms */
    _keypoints.clear();
    _keypoints.reserve(n_keypoints); 
    for (int i = 0; i < n_keypoints; i++)
    {
        float x = kpts[i][1].item<float>(), y = kpts[i][0].item<float>();
        float conf = mProb[kpts[i][0]][kpts[i][1]].item<float>();
        _keypoints.push_back(cv::KeyPoint(cv::Point((int)x, (int)y), 1.0, 0.0, conf));
    }
    
    mProb.reset();
    mDesc.reset();
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

} //END namespace SuperPointSLAM