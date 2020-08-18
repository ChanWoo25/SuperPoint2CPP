#include "test.hpp"

int main()
{
    using namespace NAMU_TEST;

    // std::cout << *typeid(float(CV_PI)).name() << std::endl;
    // Output : f

    printSection(1, "cuda available");
    bool use_cuda = torch::cuda::is_available();
    std::cout << "torch::cuda::is_available()\n";
    std::cout << "My Device Type is " << 
                (use_cuda ? "Cuda!" : "CPU!") << std::endl;

    torch::Tensor tensor = torch::rand({2, 3}).cuda();
    // std::cout << tensor << std::endl;
    tensor.print();


}