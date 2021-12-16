#include "Trainer_MD.h"
#include <eigen3/Eigen/Dense>
#include "Network.h"
#include <cmath>


Trainer_MD::Trainer_MD(std::vector<std::vector<double>> training_sample, std::vector<std::vector<double>> training_labels, float lr, bool logging){
    samples = training_sample;
    labels = training_labels;
    log_mode = logging;
    learning_rate = lr;
}

Eigen::VectorXd Trainer_MD::cce_loss_ls_bf(Eigen::VectorXd prediction, std::vector<double> ground_truth,long double *running_loss){
    long double cce_loss = 0;
    int cnt = 0;
    Eigen::VectorXd loss_grad(ground_truth.size());
    for(auto cur_sampl:ground_truth){
        double x = cur_sampl * log(prediction[cnt]);
        cce_loss += cur_sampl * log(prediction[cnt]);
        //loss_grad[cnt] = - cur_sampl / prediction[cnt];
        loss_grad[cnt] =  prediction[cnt] - cur_sampl;
        cnt += 1;
    }
    *running_loss += -cce_loss;
    return loss_grad;
}

Network Trainer_MD::train_epochs(int n_epochs, Network to_be_trained){
    Network out_net = to_be_trained;
    for (int i = 1; i < n_epochs + 1; i++){
        long double epoch_loss = 0.0;
        int cnt = 0;

        for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            Eigen::VectorXd net_predict = out_net.forward(sample);
            Eigen::VectorXd loss_grad = cce_loss_ls_bf(net_predict, labels[cnt], &epoch_loss);

            out_net.backward(loss_grad);
            out_net.update_weights(learning_rate);
            cnt++;
        }
        if (log_mode == true) {
            std::cout << "Epoch: " << i << " Average Loss: " << epoch_loss / double(cnt) << std::endl;
        }    
    }
    return out_net;
}

float Trainer_MD::full_predict(Network pred_net) {
    float cnt = 0;
    long double epoch_loss = 0;

    if (log_mode == true) std::cout << "Full prediction on all training data:" << std::endl;

    for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            Eigen::VectorXd net_predict = pred_net.forward(sample);
            Eigen::VectorXd loss_grad = cce_loss_ls_bf(net_predict, labels[cnt], &epoch_loss);

            if (log_mode == true){
                std::cout <<  " Sample: " << labels[cnt].data() << std::endl;
                std::cout <<  " Prediciton: " <<  net_predict << std::endl;
            } 
            cnt++;
    }
    return epoch_loss/cnt;
}
