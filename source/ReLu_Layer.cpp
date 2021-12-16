#include "ReLu_Layer.h"

ReLu_Layer::ReLu_Layer(int in_dim){
    input_dim = in_dim;
}
//Implementation of the ReLu function for each element in a vector
Eigen::VectorXd ReLu_Layer::forward(Eigen::VectorXd input){
    cache_in = input;
    Eigen::VectorXd output = input;
    for (int j = 0; j < input_dim; j++) {
        if(input(j) < 0) {
            output(j) = 0;
        }
    }
    //std::cout << output << std::endl;
    return output;
}
//The gradient wrt. to the inputs is 0 when the input of the layer for this element was negative
Eigen::VectorXd ReLu_Layer::backward(Eigen::VectorXd prev_grad){
    Eigen::VectorXd output = prev_grad;
    for (int j = 0; j < input_dim; j++) {
        if(cache_in(j) < 0) {
            output(j) = 0;
        }
    }
    return output;
}
