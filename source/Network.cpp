#include "Network.h"


Network::Network(std::vector<Layer*> all_lyr){
    Network::layers = all_lyr;
}

//Iterate over all layers and pass he outputs of the previous layer into the next
Eigen::VectorXd Network::forward(Eigen::VectorXd input){
    Eigen::VectorXd stage_val = input;
    for(auto const& ptr_cur_layer:layers) {
        stage_val = ptr_cur_layer->forward(stage_val);
    }
    return stage_val;
}

//Iterate backwards and bass the grad wrt. to the inputs to the layer before
void Network::backward(Eigen::VectorXd loss_grad){
    Eigen::VectorXd stage_val = loss_grad;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        stage_val = (*it)->backward(stage_val);
    }
}

//Update the weights for all layers
void Network::update_weights(float lr){
    for(auto const& ptr_cur_layer:layers) {
        ptr_cur_layer->update_weights(lr);
    }
}