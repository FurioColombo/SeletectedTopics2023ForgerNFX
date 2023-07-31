/*
 ____  _____ _        _
| __ )| ____| |      / \
|  _ \|  _| | |     / _ \
| |_) | |___| |___ / ___ \
|____/|_____|_____/_/   \_\
http://bela.io
*/

#include <Bela.h>
#include "../RTNeural/RTNeural/RTNeural.h"
#include <cmath>
#include <iostream>
#include "../Bela/libraries/Gui/Gui.h"
#include <typeinfo>

Gui gui;

int N_FRAMES = 1;
const int lstm_dim = 32;
const int frame_size = 16;
float gFrequency = 440.0;
float gPhase;
float gInverseSampleRate;

// this is the definition of the actual 
// model architecture, LSTMLayerT
// in this case 1->8->1 lstm->dense (aka linear in torch)
using ModelType = RTNeural::ModelT<float, frame_size, frame_size,
    RTNeural::LSTMLayerT<float, frame_size, lstm_dim>,
    RTNeural::DenseT<float, lstm_dim, frame_size>
    >;
ModelType gModel;

void loadModel(std::ifstream& jsonStream, ModelType& model)
{
    nlohmann::json modelJson;
    jsonStream >> modelJson;

    auto& lstm = model.get<0>();
    // note that the "lstm." is a prefix used to find the 
    // lstm data in the json file so your python
    // needs to name the lstm layer 'lstm' if you use lstm. as your prefix
    std::string prefix = "lstm.";
    // for LSTM layers, number of hidden = number of params in a hidden weight set
    // divided by 4
    // auto hidden_count = modelJson[prefix + ".weight_ih_l0"].size() / 4;
    // assert that the number of hidden units is the same as this count
    // to ensure the json file we are importing matches the model we defined.
    RTNeural::torch_helpers::loadLSTM<float> (modelJson, prefix, lstm);

    auto& dense = model.get<1>();
    std::cout << "dense in size: " << dense.in_size << std::endl;
    std::cout << "dense out size: " << dense.out_size << std::endl;

    // as per the lstm prefix, here the json needs a key prefixed with dense. 
    RTNeural::torch_helpers::loadDense<float> (modelJson, "dense.", dense);
}

bool setup(BelaContext *context, void *userData)
{   
    // init gui
    gui.setup(context->projectName);

    // Print out sample rate
    std::cout << "Sample Rate : " << context->audioSampleRate << std::endl;
    std::cout << "AudioFrames per block: " << context->audioFrames << std::endl;

    // load neural model using RTNeural
    auto modelFilePath = "model.json";
    std::cout << "Loading model from path: " << modelFilePath << std::endl;
    std::cout << "test" << std::endl;

    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);
    loadModel(jsonStream, gModel);
    gModel.reset();

    // now take the model for a test spin 
    std::cout << "testing model... " << modelFilePath << std::endl;

    /*
    std::vector<float> inputs { 0.1f, 0.2f, 0.3f, 0.4f, 
                                0.1f, 0.2f, 0.3f, 0.4f, 
                                0.1f, 0.2f, 0.3f, 0.4f, 
                                0.1f, 0.2f, 0.3f, 0.4f
                                };
    */
    float inputs[16] = {    0.1f, 0.2f, 0.3f, 0.4f, 
                        0.2f, 0.3f, 0.4f, 0.5f, 
                        0.5f, 0.6f, 0.7f, 0.8f, 
                        0.9f, 0.8f, 0.7f, 0.6f
                        };

    // float* inputs = in;
    std::cout << "input type: " << typeid(inputs).name() << std::endl;

    // std::vector<float> outputs {};
    // outputs.resize(inputs.size(), {});
    auto outputs = gModel.forward(inputs);
    auto& dense = gModel.get<1>();

    //std::cout << "output type: " << typeid(outputs).name() << std::endl;
    std::cout << "dense out type: " << typeid(dense.outs).name() << std::endl;

    for(size_t i = 0; i < frame_size; ++i)
    {
        std::cout << "in: " << inputs[i] << std::endl;
        //std::cout << "out: " << outputs << std::endl;
        auto& dense = gModel.get<1>();
        std::cout << "dense out: " << dense.outs[i] << std::endl;

    }
	return true;
    
}

void render(BelaContext *context, void *userData)
{
    
    unsigned int nFrames = context->audioFrames;

    float inputs[nFrames] = { 0 };
    float outputs[nFrames] = { 0 };

    // Read audio input
	for(unsigned int n = 0; n < nFrames; n++) {
        inputs[n] = audioRead(context, n, 0);  
	}
    outputs[0] = gModel.forward(&inputs[0]);

    auto& dense = gModel.get<1>();

    // Write audio output
	for(unsigned int n = 0; n < nFrames; n++) {
        // audioWrite(context, n, 0, outputs[n]);
        audioWrite(context, n, 0, dense.outs[n]);
	}
    // std::cout << "in: " << inputs[1] << " - out: " << dense.outs[1] << std::endl;
}

void cleanup(BelaContext *context, void *userData)
{

}
