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

Gui gui;

int N_FRAMES = 1;
const int lstm_dim = 8;
float gFrequency = 440.0;
float gPhase;
float gInverseSampleRate;

// this is the definition of the actual 
// model architecture, LSTMLayerT
// in this case 1->8->1 lstm->dense (aka linear in torch)
using ModelType = RTNeural::ModelT<float, 1, 1, RTNeural::LSTMLayerT<float, 1, lstm_dim>, RTNeural::DenseT<float, lstm_dim, 1>>;
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
    auto hidden_count = modelJson[prefix + ".weight_ih_l0"].size() / 4;
    // assert that the number of hidden units is the same as this count
    // to ensure the json file we are importing matches the model we defined.
    RTNeural::torch_helpers::loadLSTM<float> (modelJson, prefix, lstm);
  
    auto& dense = model.get<1>();
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
    std::ifstream jsonStream(modelFilePath, std::ifstream::binary);
    loadModel(jsonStream, gModel);
    gModel.reset();

    // now take the model for a test spin 
    std::vector<float> inputs {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> outputs {};
    outputs.resize(inputs.size(), {});

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        outputs[i] = gModel.forward(&inputs[i]);
        std::cout << "in " << inputs[i] << " out: " << outputs[i] << std::endl;
    }
	return true;
}

void render(BelaContext *context, void *userData)
{
    unsigned int nFrames = context->audioFrames;
    // std::vector<float> inputs {};
    // std::vector<float> outputs {};
    // input.resize(nFrames, {0});
    // outputs.resize(nFrames, {0});
    // float inputs[16];
    // float outputs[16];
    float input = 0;
    float output = 0;

    // Read audio input
	for(unsigned int n = 0; n < nFrames; n++) {
        input = audioRead(context, n, 0);  

        output = gModel.forward(&input);


        audioWrite(context, n, 0, output);

        //inputs[n] = audioRead(context, n, 0) * 0.5f;
        //outputs[n] = gModel.forward(&inputs[n]);
        //audioWrite(context, n, 0, outputs[n]);
        // use this for stereo+ signals
		// for(unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
        // 
		// }
	}
    /*
    // Process input through the model 
    outputs = gModel.forward(inputs);
    //    outputs[i] = model.forward(&inputs[i]);
    //    std::vector<float> outputs {};


    // Write audio outputs
	for(unsigned int m = 0; m < nFrames; m++) {
        audioWrite(context, m, 0, outputs[m]);

        // use this for stereo+ signals
		// for(unsigned int channel = 0; channel < context->audioOutChannels; channel++) {
        // 
		// }
	}
    */
}

void cleanup(BelaContext *context, void *userData)
{

}
