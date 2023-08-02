#include <Bela.h>
#include "../Bela/libraries/Gui/Gui.h"
#include "../RTNeural/RTNeural/RTNeural.h"
#include <cmath>
#include <iostream>
#include <typeinfo>

Gui gui;

//const auto tubeScreamModelFilePath = "tube_screamer_model.json";
const auto tubeScreamModelFilePath = "model.json";
//const auto bluesDriverModelFilePath = "blues_driver_model.json";
const auto bluesDriverModelFilePath = "model.json";
const auto RATModelFilePath = "RAT_model.json";

const int lstm_dim = 6;
const int frame_size = 1;

// this is the definition of the actual
// model architecture, LSTMLayerT
// in this case 1->8->1 lstm->dense (aka linear in torch)
using ModelType = RTNeural::ModelT<float, frame_size, frame_size,
    RTNeural::LSTMLayerT<float, frame_size, lstm_dim>,
    RTNeural::DenseT<float, lstm_dim, frame_size>
    >;
ModelType gModel;
ModelType tubeScreamerModel;
ModelType bluesDriverModel;
ModelType RATModel;
ModelType models[3] = {tubeScreamerModel, bluesDriverModel, RATModel};


void loadModel(ModelType& model, auto modelWeightsPaths)
{
    std::cout << "Loading model from path: " << modelWeightsPaths << std::endl;
    std::ifstream jsonStream(modelWeightsPaths, std::ifstream::binary);

    nlohmann::json modelJson;
    jsonStream >> modelJson;
    auto& lstm = model.get<0>();
    std::string prefix = "lstm.";
    RTNeural::torch_helpers::loadLSTM<float> (modelJson, prefix, lstm);
    auto& dense = model.get<1>();
    std::cout << "dense in size: " << dense.in_size << std::endl;
    std::cout << "dense out size: " << dense.out_size << std::endl;

    // as per the lstm prefix, here the json needs a key prefixed with dense.
    RTNeural::torch_helpers::loadDense<float> (modelJson, "dense.", dense);
    model.reset();
}

void render_bypass_fx(BelaContext *context, unsigned int nFrames, float isOn){
    // bypass fx: in = out
    float input = 0;
	for(unsigned int n = 0; n < nFrames; n++) {
        input = audioRead(context, n, 0);
        audioWrite(context, n, 0, input*isOn);
	}
}

void render_wet_fx(BelaContext *context, unsigned int nFrames, int model_idx, float isOn){

    if(isOn == 0){
        // Write audio output
        for(unsigned int n = 0; n < nFrames; n++) {
            audioWrite(context, n, 0, 0);
        }
        return;
    }
    if (frame_size==1){
        float input;
        float output;
        for(unsigned int n = 0; n < nFrames; n++) {
            input = audioRead(context, n, 0);
            output = models[model_idx].forward(&input);
            audioWrite(context, n, 0, output);
        }
    } else {
        float inputs[nFrames] = { 0 };
        float outputs[nFrames] = { 0 };
        // Read audio input
        for(unsigned int n = 0; n < nFrames; n++) {
            inputs[n] = audioRead(context, n, 0);
        }
        outputs[0] = models[model_idx].forward(&inputs[0]);

        auto& dense = models[model_idx].get<1>();

        // Write audio output
        for(unsigned int n = 0; n < nFrames; n++) {
            audioWrite(context, n, 0, dense.outs[n]);
        }
    }
}

bool setup(BelaContext *context, void *userData)
{
    // GUI
    gui.setup(context->projectName);
	gui.setBuffer('f', 2);	//Set the buffer to receive from the GUI

    // Print out sample rate
    std::cout << "Sample Rate : " << context->audioSampleRate << std::endl;
    std::cout << "AudioFrames per block: " << context->audioFrames << std::endl;

    // load neural models using RTNeural
    // Tube Screamer model
    loadModel(tubeScreamerModel, tubeScreamModelFilePath);
    // Blues Driver Model
    loadModel(bluesDriverModel, bluesDriverModelFilePath);
    // RAT Model
    loadModel(RATModel, RATModelFilePath);

    // now take the models for a test spin
    std::cout << "testing models... " << std::endl;
    float inputs[16] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.2f, 0.3f, 0.4f, 0.5f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 0.8f, 0.7f, 0.6f
    };

    for(int j=0; j<3; j++) {
        std::cout << "input type: " << typeid(inputs).name() << std::endl;

        // auto outputs = models[j].forward(inputs);
        auto& dense = models[j].get<1>();

        std::cout << "dense out type: " << typeid(dense.outs).name() << std::endl;

        for(size_t i = 0; i < frame_size; ++i)
        {
            std::cout << i << " - in: " << inputs[i] << std::endl;
            auto& dense = models[j].get<1>();
            std::cout << "dense out: " << dense.outs[i] << std::endl;

        }
    }
    std::cout << "setup complete" << std::endl;

	return true;
}

void render(BelaContext *context, void *userData)
{
    // GUI DataBuffer
	DataBuffer& buffer = gui.getDataBuffer(0);
	float* data = buffer.getAsFloat();
    float isOn = (float) data[0];
    int fxType = (int) data[1];
    unsigned int nFrames = context->audioFrames;

    switch (fxType){
        case 0:
            render_bypass_fx(context, nFrames, isOn);
            break;
        default:
            render_wet_fx(context, nFrames, fxType, isOn);
    }
}



void cleanup(BelaContext *context, void *userData)
{

}
