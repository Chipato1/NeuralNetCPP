#pragma once
#include "Network.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
/*
This class implements the gradient descent learning and later prediciton for networks with one output.
*/
class Trainer {
   public:
    Trainer(std::vector<std::vector<double>> training_sample, std::vector<double> training_labels, float lr, bool logging);
    Network train_epochs(int n_epochs, Network to_be_trained);
    float full_predict(Network pred_net);

   
   private:
    std::vector<std::vector<double>> samples;
    std::vector<double> labels;
    float learning_rate;
    bool log_mode;
};