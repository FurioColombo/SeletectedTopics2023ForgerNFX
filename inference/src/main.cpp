#include <iostream>
#include <filesystem>
#include <RTNeural/RTNeural.h>

// this is the definition of the actual 
// model architecture, LSTMLayerT
// in this case 1->8->1 lstm->dense (aka linear in torch)
using ModelType = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, 8>, RTNeural::DenseT<float, 8, 1>>;

void loadModel(std::ifstream& jsonStream, ModelType& model)
{
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& lstm = model.get<0>();
    // note that the "lstm." is a prefix used to find the 
    // lstm data in the json file so your python
    // needs to name the lstm layer 'lstm' if you use lstm. as your prefix
    std::string prefix = "lstm.";
    // for LSTM layers, number of hidden  = number of params in a hidden weight set
    // divided by 4
    auto hidden_count = modelJson[prefix + ".weight_ih_l0"].size() / 4;
    // assert that the number of hidden units is the same as this count
    // to ensure the json file we are importing matches the model we defined.
    RTNeural::torch_helpers::loadLSTM<float> (modelJson, prefix, lstm);
  
    auto& dense = model.get<1>();
    // as per the lstm prefix, here the json needs a key prefixed with dense. 
    RTNeural::torch_helpers::loadDense<float> (modelJson, "dense.", dense);
}


int main(int argc, char* argv[])
{
    auto modelFilePath = "C:/Users/Marco Furio Colombo/Desktop/Polimi - MAE/Selected Topics/Polimi_starter_v4/035c_train_lstm/furio_python/EGFx/scripts/model.json";
    assert(std::filesystem::exists(modelFilePath));

    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);
    ModelType model;
    loadModel(jsonStream, model);
    model.reset();

    // now take the model for a spin :) 
    std::vector<float> inputs {1.0, 2.0, 3.0, 4.0};
    std::vector<float> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = model.forward(&inputs[i]);
        std::cout << "in " << inputs[i] << " out: " << outputs[i] << std::endl;
    }
}
