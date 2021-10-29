#include "ReLu.h"
#include <cmath>

ReLu::ReLu(int in_dim, int out_dim) : Layer(in_dim, out_dim){}

void ReLu::init_weights(){
    double mean = 0;
    double std = std::sqrt(2.0 / static_cast<double>(ReLu::weights.size() * ReLu::weights[0].size()));
}

std::vector<double> ReLu::forward(std::vector<double> input){

}
std::vector<std::vector<double>> ReLu::backward(std::vector<double> prev_grad){
    
}