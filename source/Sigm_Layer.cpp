#include "Sigm_Layer.h"

Sigm_Layer::Sigm_Layer(int in_dim){
    input_dim = in_dim;
}

//Calculate the sigmoid function for each element in a vector
Eigen::VectorXd Sigm_Layer::forward(Eigen::VectorXd input){
    Eigen::VectorXd output = input;
    for (int j = 0; j < input_dim; j++) {
        output[j] = 1 / ( 1 + exp(-input[j]));
    }
    //std::cout << output << std::endl;
    cache_in = output;
    return output;
}

// The grad. wrt to the inputs can be computed as a function of the precious outputs
Eigen::VectorXd Sigm_Layer::backward(Eigen::VectorXd prev_grad){ 
    Eigen::VectorXd output = prev_grad;
    for (int j = 0; j < input_dim; j++) {
        output[j] = (1 - cache_in[j]) * cache_in[j] * prev_grad[j]; 
    }
    //std::cout << output << std::endl;
    return output;
}