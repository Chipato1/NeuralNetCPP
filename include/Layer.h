#pragma once
#include <vector>

class Layer {
   public:
    virtual std::vector<double> forward(std::vector<double> input);
    virtual std::vector<std::vector<double>> backward(std::vector<double> prev_grad);
    virtual void init_weights();

    //Layer(int in_dim, int out_dim);
    ~Layer();
    void update_weights(std::vector<std::vector<double>> new_weights);

   protected:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    int input_dim;
    int output_dim;
};