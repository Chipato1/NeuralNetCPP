#include "Trainer.h"


Trainer::Trainer(std::vector<std::vector<double>> training_sample, std::vector<double> training_labels, float lr, bool logging){
    samples = training_sample;
    labels = training_labels;
    log_mode = logging;
    learning_rate = lr;
}

//Train a given number of epochs
Network Trainer::train_epochs(int n_epochs, Network to_be_trained){
    std::ofstream log_file;
    log_file.open("loss_per_sample.csv", std::ofstream::trunc);
    log_file << "loss\n";
    
    std::ofstream epoch_log_file;
    epoch_log_file.open("loss_per_epoch.csv", std::ofstream::trunc);
    epoch_log_file << "epoch-loss\n";

    Network out_net = to_be_trained;
    for (int i = 1; i < n_epochs + 1; i++){
        double epoch_loss = 0;
        int cnt = 0;
        for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            float label = labels[cnt];

            Eigen::VectorXd net_predict = out_net.forward(sample);
            int loss_mode = 1;

            double loss = (double(0.5) * (double(net_predict(0)) - double(label)) * (double(net_predict(0)) - double(label)));
            epoch_loss += loss;
            log_file << loss << "\n";
            Eigen::VectorXd loss_grad(1);
            loss_grad << net_predict(0) - label;
             
            out_net.backward(loss_grad);
            out_net.update_weights(learning_rate);
            cnt++;
        }
        if (log_mode == true) {
            std::cout << "Epoch: " << i << " Average Loss: " << epoch_loss / double(cnt) << std::endl;
            epoch_log_file << epoch_loss / double(cnt) << "\n";
        }    
    }
    epoch_log_file.close();
    log_file.close();
    return out_net;
}

//Do a full prediction on the data with the passed network, return loss
float Trainer::full_predict(Network pred_net) {
    std::ofstream pred_file;
    pred_file.open("prediction.csv", std::ofstream::trunc);
    pred_file << "sample,prediction\n";
    float cnt = 0;
    float avg_loss = 0;
    if (log_mode == true) std::cout << "Full prediction on all training data:" << std::endl;
    for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            float label = labels[cnt];
            Eigen::VectorXd net_predict = pred_net.forward(sample);
            avg_loss += (float(0.5) * (float(net_predict(0)) - float(label)) * (float(net_predict(0)) - float(label)));
            if (log_mode == true) {
                std::cout <<  " Sample: " << cnt << " Label: " << label << " Pred Label: " << net_predict << std::endl;
                pred_file << label << "," << net_predict << std::endl;
            }
            cnt++;
    }
    return avg_loss/cnt;
}
