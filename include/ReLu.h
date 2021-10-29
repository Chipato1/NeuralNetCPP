#include "Layer.h"
#include <iostream>

class ReLu: public Layer {
   public:
    //ReLu(int in_dim, int out_dim);
    std::vector<double> forward(std::vector<double> input);
    std::vector<std::vector<double>> backward(std::vector<double> prev_grad);
    void init_weights();
    
   protected:
};