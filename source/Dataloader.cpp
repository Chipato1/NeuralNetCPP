#include "Dataloader.h"

/*
This class contains all functionality to load the data needed for the experiments.
The class has local attributes for all data. The data is generated in the constructor.
*/
Dataloader::Dataloader(){
    std::uniform_real_distribution<double> distribution(0, 1.0);
    std::default_random_engine generator;
    for(int i = 0; i < 500; i++){
        std::vector<double> v;
        for (int x = 0; x < 4 ; x++){
            v.push_back(distribution(generator));
        }
        sin_samples.push_back(v);
        sin_labels.push_back(sin(v[0] - v[1] + v[2] - v[3]));
    }

    //Load the Letter dataset
    std::ifstream inFile("data/letter-recognition.txt");
    std::vector<std::vector<double>> data_vec;
    std::vector<int> label_vec;

    if (inFile.is_open())
    {
        std::string line;
        while(std::getline(inFile,line) )
        {
            std::stringstream ss(line);
            std::string Label;

            //Save to label vector to an int mapping A -> 0 ... Z->25
            std::getline(ss, Label, ',');
            const char *cstr = Label.c_str();
            label_vec.push_back(int(cstr[0]) - 65);

            std::vector<double> sample;
            std::string num;

            while(std::getline(ss,num,',') ){
                 sample.push_back(std::stoi(num));
            }
            data_vec.push_back(sample);
        }
    }
    std::vector<std::vector<double>> one_hot_lab;
    for(auto cur_lab:label_vec) {
        std::vector<double> sample_oh(26, 0.0);
        sample_oh[cur_lab] = 1; 
        one_hot_lab.push_back(sample_oh);
    }
    
    letter_samples = data_vec;
    letter_labels = one_hot_lab;
}

std::vector<std::vector<double>> Dataloader::get_xor_samples(){
    std::vector<std::vector<double>> xor_samples = {{0, 0},
                                                    {0, 1},
                                                    {1, 0},
                                                    {1, 1}};
    return xor_samples;
}

std::vector<double> Dataloader::get_xor_labels(){
    std::vector<double> xor_labels = {0, 1, 1, 0};
    return xor_labels;
}

std::vector<std::vector<double>> Dataloader::get_sin_train_samples(){
    std::vector<std::vector<double>> samples(&sin_samples[0],&sin_samples[399]);
    return samples;
}

std::vector<double> Dataloader::get_sin_train_labels(){
    std::vector<double> labels(&sin_labels[0],&sin_labels[399]);
    return labels;
}

std::vector<std::vector<double>> Dataloader::get_sin_test_samples(){
    std::vector<std::vector<double>> samples(&sin_samples[400],&sin_samples[500]);
    return samples;
}

std::vector<double> Dataloader::get_sin_test_labels(){
    std::vector<double> labels(&sin_labels[400],&sin_labels[500]);
    return labels;
}


std::vector<std::vector<double>> Dataloader::get_letter_train_samples() {
    std::vector<std::vector<double>> samples(&letter_samples[0],&letter_samples[15999]);
    return samples;
}
std::vector<std::vector<double>> Dataloader::get_letter_train_labels() {
    std::vector<std::vector<double>> labels(&letter_labels[0],&letter_labels[15999]);
    return labels;
}

std::vector<std::vector<double>> Dataloader::get_letter_test_samples(){
    std::vector<std::vector<double>> samples(&letter_samples[16000],&letter_samples[20000]);
    return samples;
}

std::vector<std::vector<double>> Dataloader::get_letter_test_labels(){
    std::vector<std::vector<double>> labels(&letter_labels[16000],&letter_labels[20000]);
    return labels;
}