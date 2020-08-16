#ifndef TEST_H
#define TEST_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <typeinfo>

int main()
{
    int i(0);
    std::cout << *typeid(float(CV_PI)).name() << std::endl;
}

#endif