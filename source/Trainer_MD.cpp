#include "Trainer_MD.h"

Trainer_MD::Trainer_MD(std::vector<std::vector<double>> training_sample, std::vector<std::vector<double>> training_labels, float lr, bool logging){
    samples = training_sample;
    labels = training_labels;
    log_mode = logging;
    learning_rate = lr;
}

//This function is used locally to compute the gradient and update the running loss as pass by reference
Eigen::VectorXd Trainer_MD::loss_ls_bf(Eigen::VectorXd prediction, std::vector<double> ground_truth, long double *running_loss){
    
    long double loss = 0;
    int cnt = 0;
    Eigen::VectorXd loss_grad(ground_truth.size());
    for(auto cur_sampl:ground_truth){
        loss +=  pow(double(prediction[cnt]) - double(cur_sampl), 2);
        loss_grad[cnt] =  (prediction[cnt] - cur_sampl);// / ground_truth.size(); // Leave out the constant value as it only decreases gradient constantly all the time -> Higher learning rate
        cnt += 1;
    }
    loss = loss / cnt;
    *running_loss += loss;
    return loss_grad;
}

//train the network on the data with a given number of epochs
Network Trainer_MD::train_epochs(int n_epochs, Network to_be_trained){
    Network out_net = to_be_trained;
    std::ofstream log_file;
    log_file.open("loss_per_epoch.csv", std::ofstream::trunc);
    log_file << "epochloss\n";
    for (int i = 1; i < n_epochs + 1; i++){
        long double epoch_loss = 0.0;
        int cnt = 0;

        for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            Eigen::VectorXd net_predict = out_net.forward(sample);
            Eigen::VectorXd loss_grad = loss_ls_bf(net_predict, labels[cnt], &epoch_loss);

            out_net.backward(loss_grad);
            out_net.update_weights(learning_rate);
            cnt++;
        }
        if (log_mode == true) {
            std::cout << "Epoch: " << i << " Average Loss: " << epoch_loss / double(cnt) << std::endl;
            log_file << epoch_loss / double(cnt) << "\n";
        }    
    }
    log_file.close();
    return out_net;
}

//Do a prediciton with loss and accuracy on the hole data that was passed in the constructor
float Trainer_MD::full_predict(Network pred_net) {
    float cnt = 0;
    long double epoch_loss = 0;

    if (log_mode == true) std::cout << "Full prediction on all training data:" << std::endl;
    int correctval = 0;
    for(auto cur_sampl:samples) {
            Eigen::VectorXd sample = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(cur_sampl.data(), cur_sampl.size());
            
            Eigen::VectorXd net_predict = pred_net.forward(sample);
            Eigen::VectorXd loss_grad = loss_ls_bf(net_predict, labels[cnt], &epoch_loss);
            
            
            if (log_mode == true){

                int pred_idx = 0;
                int label_idx = 0;
                double max_pred = 0;
                double max_label = 0;
                int counter = 0;

                for(auto cur_lab:labels[cnt]){
                    if (cur_lab > max_label){
                        max_label = cur_lab;
                        label_idx = counter;
                    }

                    if (net_predict[counter] > max_pred){
                        
                        max_pred = net_predict[counter];
                        pred_idx = counter;
                    }
                    counter++;
                }
                if (log_mode == true) std::cout <<  " Sample: " <<  label_idx <<  " Prediciton: " << pred_idx << " Pred-val: " << max_pred << std::endl;
                if(label_idx == pred_idx){
                    correctval++;
                }
            } 
            cnt++;
    }
    std::cout <<  " Accuracy: " << correctval / cnt << std::endl;
    return epoch_loss / cnt;
}
