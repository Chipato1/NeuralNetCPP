#include <iostream>
#include "ReLu_Layer.h"
#include "Sigm_Layer.h"
#include "Sofm_Layer.h"
#include "Affine_Layer.h"
#include "Network.h"
#include <eigen3/Eigen/Dense>

#include <vector>
#include "Trainer.h"
#include "Trainer_MD.h"
#include "Dataloader.h"



using namespace Eigen;

int main() {
    Dataloader loader_1;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and Validate Xor Exercise
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Affine_Layer* first_layer = new Affine_Layer(2,4);
    Sigm_Layer* second_layer  = new Sigm_Layer(4);
    Affine_Layer* third_layer = new Affine_Layer(4,1);

    std::vector<Layer*> xor_net_lyrs = {first_layer, second_layer, third_layer};

    Network new_net(xor_net_lyrs);
    
    
    float lr = 1; // Init with 0, 0,8

    Trainer xor_train(loader_1.get_xor_samples(), loader_1.get_xor_labels(), lr, true);

    new_net = xor_train.train_epochs(350, new_net);
    float val_loss = xor_train.full_predict(new_net);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and Validate Sin Exercise
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    Affine_Layer* l1_s = new Affine_Layer(4,16);
    Sigm_Layer* l2_s  = new Sigm_Layer(16);
    Affine_Layer* l3_s = new Affine_Layer(16,10);
    Sigm_Layer* l4_s  = new Sigm_Layer(10);
    Affine_Layer* l5_s = new Affine_Layer(10,1);

    std::vector<Layer*> sin_net_lyrs = {l1_s, l2_s, l3_s, l4_s, l5_s};
    Network sin_net(sin_net_lyrs);
    
    Trainer sin_train(loader_1.get_sin_train_samples(), loader_1.get_sin_train_labels(), 0.1, true);
    sin_net = sin_train.train_epochs(250, sin_net);

    Trainer sin_test(loader_1.get_sin_test_samples(), loader_1.get_sin_test_labels(), 0.001, true);
    float loss_s = sin_test.full_predict(sin_net);
    std::cout << "Loss: "<< loss_s << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Train and Validate Letter Exercise
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //This is the setup for Letter experiment
    //For the optimal model choose 100 epochs, lr = 0.07, hs = 35, and add l4 and L5 in the same way as
    //For the second experiment  choose 1000 epochs, lr = 0.1, hs = 75, and delete l4 and L5
    int hs = 35;
    lr = 0.07;
    Affine_Layer* l1 = new Affine_Layer(16,hs);
    Sigm_Layer* l2  = new Sigm_Layer(hs);
    Affine_Layer* l3 = new Affine_Layer(hs,hs);
    Sigm_Layer* l4  = new Sigm_Layer(hs);
    Affine_Layer* l5 = new Affine_Layer(hs,26);
    Sigm_Layer* l6 = new Sigm_Layer(26);

    std::vector<Layer*> let_net_lyrs = { l1, l2, l3, l4, l5, l6};

    Network letter_net(let_net_lyrs);
    
    // Beware this by far the most complex experiment and it takes some time to compute
    Trainer_MD let_train(loader_1.get_letter_train_samples(), loader_1.get_letter_train_labels(), lr, true);
    letter_net = let_train.train_epochs(100, letter_net);
    float loss = let_train.full_predict(letter_net);
    std::cout << "Loss: "<< loss << std::endl;

    Trainer_MD let_test(loader_1.get_letter_test_samples(), loader_1.get_letter_test_labels(), 1, true);
    float loss1 = let_test.full_predict(letter_net);
    std::cout << "Loss1: "<< loss1 << std::endl;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Avoid memory leak
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    for(auto lay:xor_net_lyrs) {
        delete lay;
    }
    for(auto lay:sin_net_lyrs) {
        delete lay;
    }
    for(auto lay:let_net_lyrs) {
        delete lay;
    }

    return 0;
}