#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>

class Layer {
    public:
    virtual Eigen::VectorXd forward(Eigen::VectorXd input) = 0;
    virtual Eigen::VectorXd backward(Eigen::VectorXd prev_grad)  = 0;
    virtual void update_weights(float lr) = 0;
};