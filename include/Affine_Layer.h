#pragma once
#include "Layer.h"
#include <vector>
#include <eigen3/Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

/*
A Layer that performs the function Y = Wx + B.
The weights and biases can be updated, the local gradients are stored in the class
*/
class Affine_Layer : public Layer{
   public:
    Eigen::VectorXd forward(Eigen::VectorXd input);
    Eigen::VectorXd backward(Eigen::VectorXd prev_grad);
    Affine_Layer(int in_dim, int out_dim);

    void update_weights(float lr);
    void init_weights();

   //protected:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;

    Eigen::MatrixXd w_gd;
    Eigen::VectorXd b_gd;
    Eigen::VectorXd cache_in;
    int input_dim;
    int output_dim;
};