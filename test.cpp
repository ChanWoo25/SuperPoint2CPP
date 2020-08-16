//#include "test.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <typeinfo>
int main()
{
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
}
