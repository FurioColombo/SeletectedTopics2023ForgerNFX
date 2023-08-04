#include <Bela.h>
#include "../Bela/libraries/Gui/Gui.h"
#include "../RTNeural/RTNeural/RTNeural.h"
#include <cmath>
#include <iostream>
#include <typeinfo>

Gui gui;

// Model paths
const auto tubeScreamerModelFilePath = "TubeScreamerNeck_egfx_bl1lstm7.json";
const auto bluesDriverModelFilePath = "BluesDriverNeck_egfx_bl1lstm7.json";
const auto RATModelFilePath = "RATNeck_egfx_bl1lstm7.json";

// Fx indexes
const int CLEAN_FX_IDX = 0;
const int TUBE_SCREAMER_FX_IDX = 1;
const int BLUES_DRIVER_FX_IDX = 2;
const int RAT_FX_IDX = 3;

// Net parameters
const int lstm_dim = 6;
const int frame_size = 1;

// Model architecture definition
using ModelType = RTNeural::ModelT<float, frame_size, frame_size,
    RTNeural::LSTMLayerT<float, frame_size, lstm_dim>,
    RTNeural::DenseT<float, lstm_dim, frame_size>
    >;
ModelType tubeScreamerModel;
ModelType bluesDriverModel;
ModelType RATModel;

// load trained model in RTNeural model
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

// filter input audio using net and play it
void _render_wet_fx(BelaContext *context, unsigned int nFrames, ModelType &model)
{
    float input = 0;
    float output = 0;
    if (frame_size == 1){
        for(unsigned int n = 0; n < nFrames; n++) {
            input = audioRead(context, n, 0);
            output = model.forward(&input);
            audioWrite(context, n, 0, output);
        }
    } else {
        float inputs[nFrames] = { 0 };
        float outputs[nFrames] = { 0 };
        // Read audio input
        for(unsigned int n = 0; n < nFrames; n++) {
            inputs[n] = audioRead(context, n, 0);
        }
        outputs[0] = model.forward(&inputs[0]);
        auto& dense = model.get<1>();

        // Write audio output
        for(unsigned int n = 0; n < nFrames; n++) {
            audioWrite(context, n, 0, dense.outs[n]);
        }
    }
}

// play plain input audio
void _render_clean_fx(BelaContext *context, unsigned int nFrames)
{
    float input;
    for(unsigned int n = 0; n < nFrames; n++) {
        input = audioRead(context, n, 0);
        audioWrite(context, n, 0, input);
    }
}

// wrapper function for playing audio according to the gui specifics
void render_fx(BelaContext *context, int modelIdx, float isOn)
{
    unsigned int nFrames = context->audioFrames;
    // if pedal is OFF, return silence
    if(isOn == 0){
        // Write audio output
        for(unsigned int n = 0; n < nFrames; n++) {
            audioWrite(context, n, 0, 0);
        }
    } else {
        switch(modelIdx) {
            case CLEAN_FX_IDX:
                _render_clean_fx(context, nFrames);
                break;
            case TUBE_SCREAMER_FX_IDX:
                _render_wet_fx(context, nFrames, tubeScreamerModel);
                break;
            case BLUES_DRIVER_FX_IDX:
                _render_wet_fx(context, nFrames, bluesDriverModel);
                break;
            case RAT_FX_IDX:
                _render_wet_fx(context, nFrames, RATModel);
                break;
        }
    }
    return;
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
    loadModel(tubeScreamerModel, tubeScreamerModelFilePath);
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

    std::cout << "input type: " << typeid(inputs).name() << std::endl;

    // auto outputs = models[j].forward(inputs);
    auto& dense = bluesDriverModel.get<1>();

    std::cout << "dense out type: " << typeid(dense.outs).name() << std::endl;

    for(size_t i = 0; i < frame_size; ++i)
    {
        std::cout << i << " - in: " << inputs[i] << std::endl;
        auto& dense = bluesDriverModel.get<1>();
        std::cout << "dense out: " << dense.outs[i] << std::endl;
    }
    std::cout << "setup complete" << std::endl;

	return true;
}

void render(BelaContext *context, void *userData)
{
    // GUI DataBuffer
	DataBuffer& buffer = gui.getDataBuffer(0);
	float* data = buffer.getAsFloat();
    float isOn = (int) data[0];
    int fxType = (int) data[1];
    render_fx(context, fxType, isOn);
}



void cleanup(BelaContext *context, void *userData)
{

}