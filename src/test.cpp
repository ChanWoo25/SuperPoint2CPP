#include <test.hpp>

namespace NAMU_TEST
{

// #########################  DEFINITION  ###############################

/*
    A distinction is made between three kinds of persistent data 
that may be associated with a Module:

    1. Parameters: tensors that record gradients, typically weights 
    updated during the backward step (e.g. the weight of a Linear module),

    2. Buffers: tensors that do not record gradients, typically updated 
    during the forward step, such as running statistics (e.g. mean and variance in the BatchNorm module),

    3. Any additional state, not necessarily tensors, 
    required for the implementation or configuration of a Module.

    Lastly, registered parameters and buffers are handled specially during a clone() operation,
which performs a deepcopy of a cloneable Module hierarchy.

*/
cv::Mat argmin_cv_mat(const cv::Mat& mat, int axis){
    if(mat.empty())
        return cv::Mat(0, 0, CV_32F);
    
    
    if(axis == 0)
    {
        int n = mat.size().height;
        int m = mat.size().width;
        cv::Mat ret(n, 1, CV_32S);
        auto ret_ptr = ret.ptr<int>(0);
        for(int i=0; i<n; i++)
        {
            auto ptr = mat.ptr<float>(i);
            float val = std::numeric_limits<float>::max();
            int idx(-1);
            for(int j=0; j<m; j++)
            {
                if(val > *(ptr++))
                    val = *ptr, idx = j;
            }
            *(ret_ptr) = idx;
        }
    }
    else if(axis == 1)
    {
        cv::Mat _mat = mat.clone().t();
        int n = _mat.size().height;
        int m = _mat.size().width;
        cv::Mat ret(n, 1, CV_32S);
        auto ret_ptr = ret.ptr<int>(0);
        for(int i=0; i<n; i++)
        {
            auto ptr = _mat.ptr<float>(i);
            float val = std::numeric_limits<float>::max();
            int idx(-1);
            for(int j=0; j<m; j++)
            {
                if(val > *(ptr++))
                    val = *ptr, idx = j;
            }
            *(ret_ptr) = idx;
        }
    }
    else
    {
        std::cerr << "Invalid axis.\n";
        exit(3);
    }
}

cv::Mat argmin_cv_mat_with_score(const cv::Mat& mat, int axis, cv::Mat& score){
    if(mat.empty())
        return cv::Mat(0, 0, CV_32F);
    
    
    if(axis == 0)
    {
        int n = mat.size().height;
        int m = mat.size().width;
        cv::Mat ret(1, n, CV_32S);
        auto ret_ptr = ret.ptr<int>(0);
        auto score_ptr = score.ptr<float>(0); 
        for(int i=0; i<n; i++)
        {
            auto ptr = mat.ptr<float>(i);
            float val = std::numeric_limits<float>::max();
            int idx(-1);
            for(int j=0; j<m; j++)
            {
                if(val > *(ptr++))
                    val = *ptr, idx = j;
            }
            *(ret_ptr++) = idx;
            *(score_ptr++) = val;
        }
    }
    else if(axis == 1)
    {
        cv::Mat _mat = mat.clone().t();
        int n = _mat.size().height;
        int m = _mat.size().width;
        cv::Mat ret(1, n, CV_32S);
        auto ret_ptr = ret.ptr<int>(0);
        auto score_ptr = score.ptr<float>(0); 
        for(int i=0; i<n; i++)
        {
            auto ptr = _mat.ptr<float>(i);
            float val = std::numeric_limits<float>::max();
            int idx(-1);
            for(int j=0; j<m; j++)
            {
                if(val > *(ptr++))
                    val = *ptr, idx = j;
            }
            *(ret_ptr++) = idx;
            *(score_ptr++) = val;
        }
    }
    else
    {
        std::cerr << "Invalid axis.\n";
        exit(3);
    }
}

SuperPoint::SuperPoint()
{
    /* 
        A Module is registered as a submodule to another Module 
        by calling register_module(), typically from within a parent 
        module’s constructor.
    */

    //SHARED ENCODER
    conv1a = register_module("conv1a", Conv2d(Conv2dOptions(1, c1, 3).stride(1).padding(1)));
    conv1b = register_module("conv1b", Conv2d(Conv2dOptions(c1, c1, 3).stride(1).padding(1)));

    conv2a = register_module("conv2a", Conv2d(Conv2dOptions(c1, c2, 3).stride(1).padding(1)));
    conv2b = register_module("conv2b", Conv2d(Conv2dOptions(c2, c2, 3).stride(1).padding(1)));

    conv3a = register_module("conv3a", Conv2d(Conv2dOptions(c2, c3, 3).stride(1).padding(1)));
    conv3b = register_module("conv3b", Conv2d(Conv2dOptions(c3, c3, 3).stride(1).padding(1)));

    conv4a = register_module("conv4a", Conv2d(Conv2dOptions(c3, c4, 3).stride(1).padding(1)));
    conv4b = register_module("conv4b", Conv2d(Conv2dOptions(c4, c4, 3).stride(1).padding(1)));

    //DETECTOR
    convPa = register_module("convPa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convPb = register_module("convPb", Conv2d(Conv2dOptions(c5, 65, 1).stride(1).padding(0)));

    //DESCRIPTOR
    convDa = register_module("convDa", Conv2d(Conv2dOptions(c4, c5, 3).stride(1).padding(1)));
    convDb = register_module("convDb", Conv2d(Conv2dOptions(c5, d1, 1).stride(1).padding(0)));
}

void SuperPoint::forward(torch::Tensor x, torch::Tensor& Prob, torch::Tensor& Desc)
{
    //SHARED ENCODER
    std::cout << "\nSHARED - ";
    x = relu(conv1a->forward(x));
    x = relu(conv1b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv2a->forward(x));
    x = relu(conv2b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv3a->forward(x));
    x = relu(conv3b->forward(x));
    x = max_pool2d(x, 2, 2);

    x = relu(conv4a->forward(x));
    x = relu(conv4b->forward(x));

    //DETECTOR
    std::cout << "DETECTOR - ";
    auto cPa = relu(convPa->forward(x));
    auto semi = convPb->forward(cPa); // [B, 65, H/8, W/8]

    //DESCRIPTOR
    std::cout << "DESCRIPTOR - ";
    auto cDa = relu(convDa->forward(x));
    auto desc = convDb->forward(cDa); // [B, 256, H/8, W/8]

    auto dn = norm(desc, 2, 1);
    desc = desc.div(unsqueeze(dn, 1));

    std::cout << "POST - ";
    semi = softmax(semi, 1);
    semi = semi.slice(1, 0, 64);
    semi = semi.permute({0, 2, 3, 1}); // [B, H/8, W/8, 64]

    int Hc = semi.size(1);
    int Wc = semi.size(2);
    std::cout << "VIEW - ";
    semi = semi.contiguous().view({-1, Hc, Wc, 8, 8});
    semi = semi.permute({0, 1, 3, 2, 4});
    semi = semi.contiguous().view({-1, Hc * 8, Wc * 8}); // [B, H, W]

    std::cout << "COPY - ";
    Prob = semi;
    Desc = desc;

    std::cout << "END - ";
}

void display()
{ // Display some information.
    // 1. Cuda Availability. && 2. GPU number.
    // 3. cudnn availability.

    bool use_cuda = torch::cuda::is_available();
    std::cout << "torch::cuda::is_available()\n";
    std::cout << "My Device Type is " << (use_cuda ? "Cuda!" : "CPU!") << std::endl;

    torch::Tensor tensor = torch::rand({2, 3}).cuda();
    // std::cout << tensor << std::endl;
    tensor.print();
    std::cout << "I have " << torch::cuda::device_count() << " GPUs.\n";
    std::cout << "cudnn is " << (torch::cuda::cudnn_is_available() ? "available" : "unavailable") << std::endl;
}
void display(std::shared_ptr<SuperPoint> net)
{
    // Display Module and Submodule's detail informations.
    // Whether it is trainable && module's name.
    std::cout.setf(std::ios::left);
    std::cout << "\n[ " << net->name() << " ]\n";
    std::cout << std::setw(12) << "Trainable" << (net->is_training() ? "On\n\n" : "Off\n\n");

    //auto subnets = net->children();
    for (auto subnet : net->children())
    {
        subnet->pretty_print(std::cout);
        std::cout << '\n';
        std::cout << std::setw(12) << "Trainable"
                    << (subnet->is_training() ? "On\n\n" : "Off\n\n");
    }
}

SuperPointFrontend::SuperPointFrontend(std::string _weight_dir, bool _use_cuda)
{
    model = std::make_shared<SuperPoint>();
    torch::load(model, _weight_dir);

    device_type = (use_cuda) ? kCUDA : kCPU;
    c10::Device device(device_type);

    tensor_opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .layout(torch::kStrided)
                        .device(device)
                        .requires_grad(false);
    if (_use_cuda)
        model->to(device);
    model->eval();
}

cv::Mat SuperPointFrontend::detect(cv::Mat &img)
{
    device_type = (use_cuda) ? kCUDA : kCPU;
    c10::Device device(device_type);

    auto x = torch::from_blob(img.clone().data, {1, 1, img.rows, img.cols}, tensor_opts);
    x /= 255;
    //std::cout << "Successfully get bolb image.\n";
    torch::Tensor prob, desc;
    model->forward(x, prob, desc);
    prob = prob.squeeze(0);
    mProb = prob.clone();// [H, W] [120, 160]
    mDesc = desc.clone();// [1, 256, H/8, W/8]
    //std::cout << "Successfully forward image.\n";
    // Nonzero인 좌표를 Tensor로 저장.
    // [H, W]에서 Threshold 이상인 픽셀은 1, otherise 0.
    //std::cout << prob << std::endl;
    auto kpts = (prob > conf_thres);
    //  at::nonzero(at::Tensor input)
    //      return the coordinates of nonzero value pixels of 'input' Tensor.
    kpts = at::nonzero(kpts); // [n_keypoints, 2]  (y, x)

    auto fkpts = kpts.to(kFloat);

    auto grid = torch::zeros({1, 1, kpts.size(0), 2}).to(device);   // [1, 1, n_keypoints, 2]

    //  Tensor.slice is the alternative function of Python's Slicing syntex. 
    //  and quite similar for using.
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / prob.size(1) - 1; // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / prob.size(0) - 1; // y

    //  [Not Perfect]
    //  at::grid_sampler(Tensor input, Tensor grid, int64 interpolation_mode, int64 padding_mode, bool align_corner)
    //
    //      Given an input and a flow-field grid, computes the output  
    //      using input values and pixel locations from grid.
    //      'input' is 4d or 5d input (N, C, Hin, Win) or (N, C, Depth, Hin, Win)
    //      according to 'input' shape determined as (N, Hout, Wout, 2) or (N, D, Hout, Wout)
    //      Resuling in 'output' (N, C, Hout, Wout) or (N, C, Depth, Hout, Wout).
    //      
    //      interpolation_mode  --  '0': bilinear, '1': nearest
    //      padding_mode        --  '0': zeros, '1': border, '2': reflection
    //      
    //  grid[0][0] 에는 keypoints 개수만큼의 (x, y)좌표가 들어있다.
    desc = at::grid_sampler(desc, grid, 0, 0, false); // [1, 256, 1, n_keypoints]       //CHANGED
    desc = desc.squeeze(0).squeeze(1);                // [256, n_keypoints]
    //std::cout << "Successfully grid sample.\n";

    // normalize to dim 1 with 2-Norm.
    // 각 키포인트에 대해서 Normalize.
    auto dn = norm(desc, 2, 1);        // [256]
    desc = desc.div(unsqueeze(dn, 1)); // [256, n_keypoints]

    desc = desc.transpose(0, 1).contiguous(); // [n_keypoints, 256]

    // After processing, back to CPU only descriptor
    if (use_cuda)
        desc = desc.to(kCPU);

    //  Convert descriptor 
    //  From at::Tensor To cv::Mat
    auto desc_size = cv::Size(desc.size(1), desc.size(0));  // [256, n_keypoints]
    int n_keypoints = desc.size(0);
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

void SuperPointFrontend::fast_nms(cv::Mat& desc_no_nms, cv::Mat& desc_nms, int img_width, int img_height)
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

void SuperPointFrontend::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint> &keypoints, bool nms)
{
    auto prob = mProb.slice(0, iniY, maxY).slice(1, iniX, maxX); // [h, w]
    auto kpts = (prob > threshold);
    kpts = nonzero(kpts); // [n_keypoints, 2]  (y, x)

    std::vector<cv::KeyPoint> keypoints_no_nms;
    for (int i = 0; i < kpts.size(0); i++)
    {
        float response = prob[kpts[i][0]][kpts[i][1]].item<float>();
        keypoints_no_nms.push_back(cv::KeyPoint(kpts[i][1].item<float>(), kpts[i][0].item<float>(), 8, -1, response));
    }

    if (nms)
    {
        cv::Mat conf(keypoints_no_nms.size(), 1, CV_32F);
        std::cout << "SPDetector::getKeyPoints : conf's size: " << conf.size() << std::endl;

        for (size_t i = 0; i < keypoints_no_nms.size(); i++)
        {
            int x = keypoints_no_nms[i].pt.x;
            int y = keypoints_no_nms[i].pt.y;
            conf.at<float>(i, 0) = prob[y][x].item<float>();
        }

        // cv::Mat descriptors;

        int border = 0;
        int dist_thresh = 4;
        int height = maxY - iniY;
        int width = maxX - iniX;

        // Keypoints의 좌표를 담은 [n, 2] 벡터와 Confidence를 담은 [n, 1] 벡터.
        NMS2(keypoints_no_nms, conf, keypoints, border, dist_thresh, width, height);
    }
    else
    {
        keypoints = keypoints_no_nms;
    }
}

void SuperPointFrontend::computeDescriptors(cv::Mat& descriptors) //const std::vector<cv::KeyPoint> &kpts_nms, cv::Mat &desc_nms
{
    cv::Mat kpt_mat(kpts_nms.size(), 2, CV_32F); // [n_keypoints, 2]  (y, x)

    auto kpt_mat_ptr = kpt_mat.ptr<float>(0);
    for (auto iter = kpts_nms.begin(); iter != kpts_nms.end(); iter++)
    {
        *(kpt_mat_ptr++) = (float)(*iter).pt.y;
        *(kpt_mat_ptr++) = (float)(*iter).pt.x;
    }
    
    auto fkpts = torch::from_blob(kpt_mat.data, {(long long)kpts_nms.size(), 2}, kFloat); //CHANGED (long long)

    auto grid = torch::zeros({1, 1, fkpts.size(0), 2});                         // [1, 1, n_keypoints, 2]
    grid[0][0].slice(1, 0, 1) = 2.0 * fkpts.slice(1, 1, 2) / mProb.size(1) - 1; // x
    grid[0][0].slice(1, 1, 2) = 2.0 * fkpts.slice(1, 0, 1) / mProb.size(0) - 1; // y

    auto desc = grid_sampler(mDesc, grid, 0, 0, false); // [1, 256, 1, n_keypoints]         //CHANGED ,false
    desc = desc.squeeze(0).squeeze(1);                  // [256, n_keypoints]

    // normalize to 1
    auto dn = norm(desc, 2, 1);
    desc = desc.div(unsqueeze(dn, 1));

    desc = desc.transpose(0, 1).contiguous(); // [n_keypoints, 256]
    desc = desc.to(kCPU);

    descriptors = cv::Mat(cv::Size(desc.size(1), desc.size(0)), CV_32FC1, desc.data_ptr<float>());
}

void SuperPointFrontend::NMS2
    (std::vector<cv::KeyPoint> det, cv::Mat conf, std::vector<cv::KeyPoint> &pts,
     int border, int dist_thresh, int img_width, int img_height)
{

    std::vector<cv::Point2f> pts_raw;

    for (int i = 0; i < det.size(); i++)
    {

        int u = (int)det[i].pt.x;
        int v = (int)det[i].pt.y;

        pts_raw.push_back(cv::Point2f(u, v));
    }

    cv::Mat grid = cv::Mat(cv::Size(img_width, img_height), CV_8UC1);
    cv::Mat inds = cv::Mat(cv::Size(img_width, img_height), CV_16UC1);

    cv::Mat confidence = cv::Mat(cv::Size(img_width, img_height), CV_32FC1);

    grid.setTo(0);
    inds.setTo(0);
    confidence.setTo(0);

    for (int i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int)pts_raw[i].x;
        int vv = (int)pts_raw[i].y;

        grid.at<char>(vv, uu) = 1;
        inds.at<unsigned short>(vv, uu) = i;

        confidence.at<float>(vv, uu) = conf.at<float>(i, 0);
    }

    cv::copyMakeBorder(grid, grid, dist_thresh, dist_thresh, dist_thresh, dist_thresh, cv::BORDER_CONSTANT, 0);

    for (int i = 0; i < pts_raw.size(); i++)
    {
        int uu = (int)pts_raw[i].x + dist_thresh;
        int vv = (int)pts_raw[i].y + dist_thresh;

        if (grid.at<char>(vv, uu) != 1)
            continue;

        for (int k = -dist_thresh; k < (dist_thresh + 1); k++)
            for (int j = -dist_thresh; j < (dist_thresh + 1); j++)
            {
                if (j == 0 && k == 0)
                    continue;

                if (confidence.at<float>(vv + k, uu + j) < confidence.at<float>(vv, uu))
                    grid.at<char>(vv + k, uu + j) = 0;
            }
        grid.at<char>(vv, uu) = 2;
    }

    size_t valid_cnt = 0;
    std::vector<int> select_indice;

    for (int v = 0; v < (img_height + dist_thresh); v++)
    {
        for (int u = 0; u < (img_width + dist_thresh); u++)
        {
            if (u - dist_thresh >= (img_width - border) || u - dist_thresh < border || v - dist_thresh >= (img_height - border) || v - dist_thresh < border)
                continue;

            if (grid.at<char>(v, u) == 2)
            {
                int select_ind = (int)inds.at<unsigned short>(v - dist_thresh, u - dist_thresh);
                cv::Point2f p = pts_raw[select_ind];
                float response = conf.at<float>(select_ind, 0);
                pts.push_back(cv::KeyPoint(p, 8.0f, -1, response));

                select_indice.push_back(select_ind);
                valid_cnt++;
            }
        }
    }

    // descriptors.create(select_indice.size(), 256, CV_32F);

    // for (int i=0; i<select_indice.size(); i++)
    // {
    //     for (int j=0; j < 256; j++)
    //     {
    //         descriptors.at<float>(i, j) = desc.at<float>(select_indice[i], j);
    //     }
    // }
}

void printSection(int n, std::string s)
{
    std::cout << " < TEST Section " << n << " > \n"
                << "--Test about " << s << "--\n\n";
}



cv::Mat VideoStreamer::read_image(const string& path)
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

Tracker::Tracker()
{

}

cv::Mat Tracker::nn_match_two_way(const cv::Mat& desc1, const cv::Mat& desc2)
{
    assert(desc1.size() == desc2.size() && nn_thres > 0.0);
    
    // Compute L2 Distance. Easy since vectors are unit normalized.
    cv::Mat dmat = desc1.t() * desc2;
    dmat = cv::min(cv::max(dmat, -1), 1);
    cv::sqrt((2 - 2 * dmat), dmat);

    // Get NN indices and scores.
    int n = desc1.size().width;
    cv::Mat score(1, n, CV_32F);
    cv::Mat idx = argmin_cv_mat_with_score(dmat, 1, score);
    cv::Mat keep = (score < nn_thres); // return type -- CV_8U '255 or 0'

    cv::Mat idx2 = argmin_cv_mat(dmat, 0);
    cv::Mat keep_bi(1, n, CV_8U);
    // ptr로 불러올 때는 C++의 Type을 명시해 주어야 한다. CV_로 시작하는 타입을 < > 사이에 입력시 에러뜸.
    auto idx2_ptr = idx2.ptr<int>(0);
    auto keep_bi_ptr = keep_bi.ptr<u_char>(0);
    for(int i = 0; i<n; i++)
        *(keep_bi_ptr++) = (i == idx2.at<int>(idx.at<int>(i, 0), 0) ? 255 : 0);
    keep = min(keep, keep_bi);
    
    int k(0);
    for(int i=0 ;i<n;i++)
        if(*(keep_bi_ptr++) > 0) k++;
    
    keep_bi_ptr = keep_bi.ptr<u_char>(0);
    cv::Mat m_idx1, m_idx2, m_score;
    for(int i=0 ;i<n;i++)
    {
        if(*(keep_bi_ptr++) > 0)
        {
            m_idx1.push_back(i);
            m_idx2.push_back(idx.at<int>(i, 0));
            m_score.push_back(score.at<float>(i, 0));
        }
    }
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

void test_with_magicleap(){
    cv::namedWindow("linux", cv::WINDOW_NORMAL);
    cv::Mat gray_img = cv::imread("../Dataset/magicleap.png", cv::IMREAD_GRAYSCALE);
    if(gray_img.empty()){
        std::cout << " No Image.\n";
        exit(1);
    }
    gray_img.convertTo(gray_img, CV_32FC1);
    cv::imshow("linux", gray_img);
    auto key = cv::waitKey(0);
    // cv::imshow("linux", gray_img);
    std::cout << cv_type2str(gray_img.type()) << std::endl;
}



} // Namespace NAMU_TEST END
