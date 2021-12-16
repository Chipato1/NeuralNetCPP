#include "Sofm_Layer.h"


Sofm_Layer::Sofm_Layer(int in_dim){
    input_dim = in_dim;
}

//compue the sofm function for each value in the vector
Eigen::VectorXd Sofm_Layer::forward(Eigen::VectorXd input){
    Eigen::VectorXd output = input;
    double sum_e = 0; 
    for (int j = 0; j < input_dim; j++) {
        sum_e += exp(input[j]);
    }

    for (int j = 0; j < input_dim; j++) {
        output[j] = exp(input[j]) / sum_e;
        if(output[j] < 0){
            int x = 0;
        }
    }
    cache_in = output;
    return output;
}

//This is commented out because one can compute the grad for sofm. and cross entropy toegter in the trainer, however it is currently not used anyways
Eigen::VectorXd Sofm_Layer::backward(Eigen::VectorXd prev_grad){
    /*Eigen::VectorXd output = prev_grad;
    for (int j = 0; j < input_dim; j++) {
        output[j] = (1 - cache_in[j]) * cache_in[j] * prev_grad[j]; 
    }*/
    return prev_grad;
}