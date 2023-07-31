/*
For sending a buffer from p5.js to Bela we use the function:
```
Bela.data.sendBuffer(0, 'float', buffer);
```
where the first argument (0) is the index of the sent buffer, second argument ('float') is the type of the
data, and third argument (buffer) is the data array to be sent.
*/
const VOLUME_SLIDER_IDX = 0

const CLEAN_BUTTON_IDX = 1
const TUBESCREAMER_BUTTON_IDX = 2
const BLUESDRIVER_BUTTON_IDX = 3
const RAT_BUTTON_IDX = 4

//buffer to send to Bela. 5 elements:
// 4 buttons [clean, TubeScreamer, BluesDriver, RAT]
// 1 slider [Volume]
let buffer = [0, 0, 0, 0];
//current state of the PLAY/STOP button.
let buttonStates = [1, 0, 0, 0];
let buttonNames = ['Clean', 'TubeScreamer', 'BluesDriver', 'RAT']

function setup() {
	//Create a thin canvas where to allocate the elements (this is not strictly neccessary because
	//we will use DOM elements which can be allocated directly in the window)
//	createCanvas(windowWidth * 3/4, windowHeight / 5);
	createCanvas(windowWidth, windowHeight);

	//Create two sliders, first to control frequency and second to control amplitude of oscillator
	//both go from 0 to 100, starting with value of 60
    volumeSliderMin = 0
    volumeSliderMax = 100
    volumeSliderInitial = 80
	volumeSlider = createSlider(volumeSliderMin, volumeSliderMax, volumeSliderInitial);

	//Create Distorsion type buttons
	cleanButton = createButton("Clean");
	cleanButton.mouseClicked( function(){
         changeButtonState(CLEAN_BUTTON_IDX, cleanButton);
        }
    );

    //Create Distorsion type buttons
	tubeScreamerButton = createButton("Tube\nScreamer");
	tubeScreamerButton.mouseClicked( function(){
        changeButtonState(TUBESCREAMER_BUTTON_IDX, tubeScreamerButton);
       }
    );
    //Create Distorsion type buttons
	bluesDriverButton = createButton("Blues\nDriver");
	bluesDriverButton.mouseClicked( function(){
        changeButtonState(BLUESDRIVER_BUTTON_IDX, bluesDriverButton);
       }
    );
    //Create Distorsion type buttons
	RATButton = createButton("RAT");
	RATButton.mouseClicked( function(){
        changeButtonState(RAT_BUTTON_IDX, RATButton);
       }
    );
	//This function will format colors and positions of the DOM elements (sliders, button and text)
	formatDOMElements();
}

function draw() {
    background(5,5,255,1);

    //store values in the buffer
	buffer[VOLUME_SLIDER_IDX]       =   volumeSlider.value()/100;
	buffer[CLEAN_BUTTON_IDX]        =   buttonStates[CLEAN_BUTTON_IDX];
    buffer[TUBESCREAMER_BUTTON_IDX] =   buttonStates[TUBESCREAMER_BUTTON_IDX];
    buffer[BLUESDRIVER_BUTTON_IDX]  =   buttonStates[BLUESDRIVER_BUTTON_IDX];
    buffer[RAT_BUTTON_IDX]          =   buttonStates[RAT_BUTTON_IDX];

	//SEND BUFFER to Bela. Buffer has index 0 (to be read by Bela),
	//contains floats and sends the 'buffer' array.
    // Bela.data.sendBuffer(0, 'float', buffer);
}

function formatDOMElements() {

	//Format Buttons and labels
    fxButtonsDimension = 100
    fxButtonsYpos = windowHeight/2 - fxButtonsDimension/2
    btnDistance = 125

    // Clean Button
    posX = fxButtonsDimension
	cleanButton.position(posX, fxButtonsYpos);
	cleanButton.size(fxButtonsDimension, fxButtonsDimension);
    setDefaultFxButtonStyle(cleanButton)

    // Tube Screamer Button
    posX = posX + btnDistance
    tubeScreamerButton.position(posX, fxButtonsYpos);
	tubeScreamerButton.size(fxButtonsDimension, fxButtonsDimension);
    setDefaultFxButtonStyle(tubeScreamerButton)

    // Blues Driver Button
    posX = posX + btnDistance
	bluesDriverButton.position(posX, fxButtonsYpos);
	bluesDriverButton.size(fxButtonsDimension, fxButtonsDimension);
    setDefaultFxButtonStyle(bluesDriverButton)

    // RAT Button
    posX = posX + btnDistance
	RATButton.position(posX, fxButtonsYpos);
	RATButton.size(fxButtonsDimension, fxButtonsDimension);
    setDefaultFxButtonStyle(RATButton)

    // Format sliders
    // Volume slider
    sliderHeight = 120
    sliderWidth = 20
    posX = posX + 200
    let back_col = color(0, 10, 10, 100);

	volumeSlider.position(posX, windowHeight/2);
    volumeSlider.size(sliderHeight, sliderWidth);
    volumeSlider.style("transform", "rotate(-90deg)");
	}

//Function that changes buttonState variable and changes button's background-color
function changeButtonState(buttonIdx, button) {
    let off_col = color(0, 10, 10, 100);
    let on_col = color(50, 110, 110, 100);
    if (buttonStates[buttonIdx] == 0){
        buttonStates[buttonIdx] = 1
        button.style('background-color', on_col);

    } else {
        buttonStates[buttonIdx] = 0
        button.style('background-color', off_col);
    }

}


//Function that changes buttonState variable and changes button's background-color
function setDefaultFxButtonStyle(button) {
	let col = color(0, 10, 10, 100);
	button.style('font-weight','bolder');
	//button.style('background-image','url(https://fg-a.com/music/maracas-animation-2018.gif)');
	button.style('border', '2px solid #000000');
	button.style('border-radius', '50%');
    button.style('background-color', col);
	button.style('color', 'white');
}
