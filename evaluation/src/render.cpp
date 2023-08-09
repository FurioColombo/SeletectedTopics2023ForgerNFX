#include <Bela.h>
#include "../Bela/libraries/Gui/Gui.h"
#include "../RTNeural/RTNeural/RTNeural.h"
#include <cmath>
#include <iostream>
#include <typeinfo>

Gui gui;

// Model paths
const auto tubeScreamerModelBl1FilePath =  "TubeScreamerNeck_egfx_bl1lstm6.json";
const auto bluesDriverModelBl1FilePath =   "BluesDriverNeck_egfx_bl1lstm6.json";
const auto RATModelBl1FilePath =           "RATNeck_egfx_bl1lstm6.json";
const auto tubeScreamerModelBl16FilePath = "TubeScreamerNeck_egfx_bl16lstm64.json";
const auto bluesDriverModelBl16FilePath =  "BluesDriverNeck_egfx_bl16lstm64.json";
const auto RATModelBl16FilePath =          "RATNeck_egfx_bl16lstm64.json";

// Fx indexes
const int CLEAN_FX_IDX = 0;
const int TUBE_SCREAMER_FX_IDX = 1;
const int BLUES_DRIVER_FX_IDX = 2;
const int RAT_FX_IDX = 3;

// Net parameters
int model_frame_size = 1;
int belaFramesPerBlock;

// Model architecture definition
using ModelTypeBl1 = RTNeural::ModelT<float, 1, 1,
    RTNeural::LSTMLayerT<float, 1, 6>,
    RTNeural::DenseT<float, 6, 1>
    >;
using ModelTypeBl16 = RTNeural::ModelT<float, 16, 16,
    RTNeural::LSTMLayerT<float, 16, 64>,
    RTNeural::DenseT<float, 64, 16>
    >;
ModelTypeBl1 tubeScreamerModelBl1;
ModelTypeBl1 bluesDriverModelBl1;
ModelTypeBl1 RATModelBl1;
ModelTypeBl16 tubeScreamerModelBl16;
ModelTypeBl16 bluesDriverModelBl16;
ModelTypeBl16 RATModelBl16;

// load trained model in RTNeural model
void _loadModelBS1(ModelTypeBl1& model, auto modelWeightsPaths)
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

// load trained model in RTNeural model
void _loadModelBS16(ModelTypeBl16& model, auto modelWeightsPaths)
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

void loadModels()
{
    // Tube Screamer model
    _loadModelBS1(tubeScreamerModelBl1, tubeScreamerModelBl1FilePath);
    // Blues Driver Model
    _loadModelBS1(bluesDriverModelBl1, bluesDriverModelBl1FilePath);
    // RAT Model
    _loadModelBS1(RATModelBl1, RATModelBl1FilePath);

    // Tube Screamer model
    _loadModelBS16(tubeScreamerModelBl16, tubeScreamerModelBl16FilePath);
    // Blues Driver Model
    _loadModelBS16(bluesDriverModelBl16, bluesDriverModelBl16FilePath);
    // RAT Model
    _loadModelBS16(RATModelBl16, RATModelBl16FilePath);
}

// filter input audio using net and play it
void _render_wet_fx(BelaContext *context, unsigned int nFrames, ModelTypeBl1 &modelBl1, ModelTypeBl16 &modelBl16)
{
    float input = 0;
    float output = 0;
    if (model_frame_size == 1){
        for(unsigned int n = 0; n < nFrames; n++) {
            input = audioRead(context, n, 0);
            output = modelBl1.forward(&input);
            audioWrite(context, n, 0, output);
        }
    } else {
        float inputs[belaFramesPerBlock] = { 0 };
        float output;

        for(int i=0; i<(int)belaFramesPerBlock/model_frame_size; i++){
            // Read audio input
            for(int n = 0; n < model_frame_size; n++) {
                inputs[n] = audioRead(context, n+i*model_frame_size, 0);
            }
            output = modelBl16.forward(inputs);
            auto& dense = modelBl16.get<1>();
            // std::cout << "dense type: " << typeid(dense).name() << std::endl;
            // std::cout << "output: " << output << std::endl;
            // std::cout << "dense: " << output << std::endl;


            // Write audio output
            for(unsigned int n = 0; n < nFrames; n++) {
                audioWrite(context, n+i*model_frame_size, 0, dense.outs[n]);
            }
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
                _render_wet_fx(context, nFrames, tubeScreamerModelBl1, tubeScreamerModelBl16);
                break;
            case BLUES_DRIVER_FX_IDX:
                _render_wet_fx(context, nFrames, bluesDriverModelBl1, bluesDriverModelBl16);
                break;
            case RAT_FX_IDX:
                _render_wet_fx(context, nFrames, RATModelBl1, RATModelBl16);
                break;
        }
    }
    return;
}

bool setup(BelaContext *context, void *userData)
{
    // GUI
    gui.setup(context->projectName);
	gui.setBuffer('f', 3);	//Set the buffer to receive from the GUI

    // Print out sample rate
    std::cout << "Sample Rate : " << context->audioSampleRate << std::endl;
    std::cout << "AudioFrames per block: " << context->audioFrames << std::endl;

    belaFramesPerBlock = context->audioFrames;
    if(context->audioFrames % model_frame_size != 0)
    {
        std::cout << "ERROR: " << belaFramesPerBlock <<
        " AudioFrames per block not compatible with model block size: "
        << model_frame_size << std::endl;
        exit(-1);
    }
    // load neural models using RTNeural
    loadModels();

	return true;
}

void render(BelaContext *context, void *userData)
{
    // GUI DataBuffer
	DataBuffer& buffer = gui.getDataBuffer(0);
	float* data = buffer.getAsFloat();
    float isOn = (int) data[0];
    int fxType = (int) data[1];
    model_frame_size = (int) data[2];
    render_fx(context, fxType, isOn);
}

void cleanup(BelaContext *context, void *userData)
{

}
