#include "Affine_Layer.h"

Affine_Layer::Affine_Layer(int in_dim, int out_dim) { 
  Affine_Layer::input_dim = in_dim;
  Affine_Layer::output_dim = out_dim;
  Affine_Layer::weights = Eigen::MatrixXd(output_dim, input_dim);
  Affine_Layer::biases = Eigen::VectorXd(output_dim);
  w_gd = Eigen::MatrixXd(output_dim, input_dim);
  b_gd = Eigen::VectorXd(output_dim);
  init_weights();
}

//Initialize weights with kaiming intialization
void Affine_Layer::init_weights() {
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0,1/(sqrt(input_dim)));
  for (int j = 0; j < output_dim; j++) {
    for (int i = 0; i < input_dim; i++) {
      weights(j,i) = distribution(generator);
    }
    biases(j) = 0 ;//distribution(generator);
  }
}

//The forward pass is a vector multiplication Wx + B
Eigen::VectorXd Affine_Layer::forward(Eigen::VectorXd input) {
  Eigen::VectorXd tmp_res = weights * input + biases;
  cache_in = input;
  return tmp_res;
}
//Compute local gradients and pass back gradient wrt to weights
Eigen::VectorXd Affine_Layer::backward(Eigen::VectorXd prev_grad) {
  for(int x = 0; x < output_dim; x++){
        for(int i = 0; i < input_dim; i++){
            w_gd(x,i) = cache_in(i) * prev_grad(x);
        }
    }
  b_gd = prev_grad;

  return weights.transpose() * prev_grad;
}
//update learnable parameters with fixed learning rate
void Affine_Layer::update_weights(float lr){
  Affine_Layer::weights = weights - lr * w_gd;
  Affine_Layer::biases = biases - lr * b_gd;
}