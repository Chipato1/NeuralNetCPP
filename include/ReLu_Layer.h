#pragma once
#include "Layer.h"
#include <random>
#include <eigen3/Eigen/Dense>
#include <iostream>

class ReLu_Layer : public Layer{
   public:
    ReLu_Layer(int in_dim);
    Eigen::VectorXd forward(Eigen::VectorXd input);
    Eigen::VectorXd backward(Eigen::VectorXd prev_grad);
    void update_weights(float lr){}

    int input_dim;
    Eigen::VectorXd cache_in;
};