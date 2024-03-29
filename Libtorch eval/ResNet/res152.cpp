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

torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size, int64_t stride=1, int64_t padding=0, bool with_bias=false)
{
  torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
  conv_options.stride(stride);
  conv_options.padding(padding);
  conv_options.bias(with_bias);
  return conv_options;
}

struct BottleNeck : torch::nn::Module {

  static const int expansion;

  int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Conv2d conv3;
  torch::nn::Sequential downsample;

  BottleNeck(int64_t inplanes, int64_t planes, int64_t stride_=1, torch::nn::Sequential downsample_=torch::nn::Sequential()):
    conv1(conv_options(inplanes, planes, 1)),
    conv2(conv_options(planes, planes, 3, stride_, 1)),
    conv3(conv_options(planes, planes * expansion , 1)),
    downsample(downsample_)
    {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv3", conv3);
    stride = stride_;
    if (!downsample->is_empty())
    {
      register_module("downsample", downsample);
    }
    }

  torch::Tensor forward(torch::Tensor x)
  {
    at::Tensor residual(x.clone());

    x = conv1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = torch::relu(x);

    x = conv3->forward(x);

    if (!downsample->is_empty()){
      residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
  }
};

const int BottleNeck::expansion = 4;


template <class Block> struct ResNet : torch::nn::Module
{

  int64_t inplanes = 64;
  torch::nn::Conv2d conv1;
  torch::nn::Sequential layer1;
  torch::nn::Sequential layer2;
  torch::nn::Sequential layer3;
  torch::nn::Sequential layer4;
  torch::nn::Linear fc;

  ResNet(int64_t num_classes=1000):
    conv1(conv_options(3, 64, 7, 2, 3)),
    layer1(_make_layer(64, 3)),
    layer2(_make_layer(128, 8, 2)),
    layer3(_make_layer(256, 36, 2)),
    layer4(_make_layer(512, 3, 2)),
    fc(512 * Block::expansion, num_classes)
    {
    register_module("conv1", conv1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
    }

  torch::Tensor forward(torch::Tensor x)
  {

    x = conv1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({x.sizes()[0], -1});
    x = fc->forward(x);

    return x;
  }


private:
  torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride=1)
  {
    torch::nn::Sequential downsample;
    if (stride != 1 or inplanes != planes * Block::expansion){
      downsample = torch::nn::Sequential(
          torch::nn::Conv2d(conv_options(inplanes, planes * Block::expansion, 1, stride))
      );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample));
    inplanes = planes * Block::expansion;
    for (int64_t i = 0; i < blocks; i++){
      layers->push_back(Block(inplanes, planes));
    }

    return layers;
  }
};

ResNet<BottleNeck> resnet152()
{
  ResNet<BottleNeck> model;
  return model;
}

int main()
{
  std::cout << "ResNet152 - CPU version" << std::endl;
  torch::Device device("cpu");
  
  int count = 1000;

  ResNet<BottleNeck> resnet = resnet152();
  resnet.to(device);
  torch::Tensor t = torch::rand({1, 3, 224, 224}).to(device);
  torch::Tensor output;

  double start, stop, duration;
  for (size_t i = 0; i < count; i++)
  {
    start = time_get();
    output = resnet.forward(t);
    stop = time_get();
    duration += (stop - start);
  }
  double latency = duration / ((double)count);

  std::cout << "For " << count << " inferences..." << std::endl;
  std::cout << "Time elapsed: " << duration << " ms." << std::endl;
  std::cout << "Time consuming: " << latency << " ms per instance." << std::endl;

  return 0;
}