// MLP

#include <torch/torch.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

double time_get()
{
    time_t time_sec;
    double nowtime=0;
    time_sec = time(NULL);
    struct timeval tmv;
    gettimeofday(&tmv, NULL);
    nowtime=time_sec*1000+tmv.tv_usec/1000;

    return nowtime;
}

struct mlp : public torch::nn::Module
{
    torch::nn::Linear fc0;
    torch::nn::Linear fc3;
    torch::nn::Linear fc6;

    mlp():
        fc0(784, 512),
        fc3(512, 128),
        fc6(128, 10)
        {
            register_module("fc0", fc0);
            register_module("fc3", fc3);
            register_module("fc6", fc6);
        }
    
    torch::Tensor forward(torch::Tensor x)
    {
        x = x.view({ -1, 28 * 28 });
        x = torch::relu(fc0->forward(x));
        x = torch::relu(fc3->forward(x));
        x = fc6->forward(x);
        
        return x;
    }

};

int main()
{
    std::cout << "MLP {1, 1, 28, 28} CPU version" << std::endl;
    mlp model;
    int count = 1000;

    torch::Tensor input = torch::rand({1, 1, 28, 28});
    torch::Tensor output;

    double start, stop, duration;
    duration = 0;
    for (size_t i = 0; i < count; i++)
    {
        start = time_get();
        output = model.forward(input);
        stop = time_get();
        duration += (stop - start);
    }
    double latency = duration / ((double)count);

    std::cout << "For " << count << " inferences..." << std::endl;
    std::cout << "Time elapsed: " << duration << " ms." << std::endl;
    std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

    return 0;
}