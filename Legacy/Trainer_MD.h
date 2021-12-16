#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>
#include "Network.h"

class Trainer_MD {
   public:
    Trainer_MD(std::vector<std::vector<double>> training_sample, std::vector<std::vector<double>> training_labels, float lr, bool logging);
    Network train_epochs(int n_epochs, Network to_be_trained);
    float full_predict(Network pred_net);

   
   private:
    Eigen::VectorXd cce_loss_ls_bf(Eigen::VectorXd prediction, std::vector<double> ground_truth, long double *running_loss);
    std::vector<std::vector<double>> samples;
    std::vector<std::vector<double>> labels;
    float learning_rate;
    bool log_mode;
};