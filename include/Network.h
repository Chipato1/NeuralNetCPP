#pragma once
#include "Layer.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>

class Network {
   public:
    Network(std::vector<Layer*>);
    Eigen::VectorXd forward(Eigen::VectorXd input);
    void backward(Eigen::VectorXd loss_grad);
    void update_weights(float lr);

   private:
    std::vector<Layer*> layers;
};