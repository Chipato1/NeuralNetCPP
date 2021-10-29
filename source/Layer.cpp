#include "Layer.h"

/*Layer::Layer(int in_dim, int out_dim) { 
  Layer::input_dim = in_dim;
  Layer::output_dim = out_dim;
  //init_weights();
}
*/
void Layer::update_weights(std::vector<std::vector<double>> new_weights){
  //TO DO check if the dimension is correct
  Layer::weights = new_weights;
}

