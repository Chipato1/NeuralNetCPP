#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

class Dataloader {
    public:
    Dataloader();
    std::vector<std::vector<double>> get_xor_samples();
    std::vector<double> get_xor_labels();

    std::vector<std::vector<double>> get_sin_train_samples();
    std::vector<double> get_sin_train_labels();
    
    std::vector<std::vector<double>> get_sin_test_samples();
    std::vector<double> get_sin_test_labels();
    
    
    std::vector<std::vector<double>> get_letter_train_samples();
    std::vector<std::vector<double>> get_letter_train_labels();
    
    std::vector<std::vector<double>> get_letter_test_samples();
    std::vector<std::vector<double>> get_letter_test_labels();
    

    private:
    std::vector<std::vector<double>> sin_samples;
    std::vector<double> sin_labels;
    std::vector<std::vector<double>> letter_samples;
    std::vector<std::vector<double>> letter_labels;
};