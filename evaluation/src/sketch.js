/*
For sending a buffer from p5.js to Bela we use the function:
```
Bela.data.sendBuffer(0, 'float', buffer);
```
where the first argument (0) is the index of the sent buffer, second argument ('float') is the type of the
data, and third argument (buffer) is the data array to be sent.
*/
let offColor;
let onColor;

const BUFFER_ONOFF_IDX = 0
const BUFFER_ACTIVE_FX_IDX = 1

const CLEAN_BUTTON_IDX = 0
const TUBESCREAMER_BUTTON_IDX = 1
const BLUESDRIVER_BUTTON_IDX = 2
const RAT_BUTTON_IDX = 3

//buffer to send to Bela. 5 elements:
// 4 buttons [clean, TubeScreamer, BluesDriver, RAT]
// 1 slider [Volume]
// buffer: [isFxOn, activeFXButton]
let buffer = [1, 0];
let activeFXButton = 0;
let isFXOn = 0;
let buttonNames = ['Clean', 'TubeScreamer', 'BluesDriver', 'RAT']

function setup() {
	createCanvas(windowWidth, windowHeight);

    offColor = color(0, 10, 10, 100);
    onColor = color(50, 110, 110, 110);

    // Create buttons
    // Create On/Off button
    onOffButton = createButton("ON/OFF");
	// Create Distorsion type buttons
	cleanButton = createButton("Clean");
	tubeScreamerButton = createButton("Tube\nScreamer");
	bluesDriverButton = createButton("Blues\nDriver");
	RATButton = createButton("RAT");
    let distorsionTypeBtns = [
        cleanButton,
        tubeScreamerButton,
        bluesDriverButton,
        RATButton
    ]

	// On/Off Button listener
    onOffButton.mouseClicked( function(){
        changeOnOffButtonState(isFXOn, onOffButton);
       }
    );

	// Distorsion type buttons listeners
    cleanButton.mouseClicked( function(){
        changeDistorsionTypeButtonState(CLEAN_BUTTON_IDX, distorsionTypeBtns);
       }
    );
    tubeScreamerButton.mouseClicked( function(){
        changeDistorsionTypeButtonState(TUBESCREAMER_BUTTON_IDX, distorsionTypeBtns);
        }
    );
    bluesDriverButton.mouseClicked( function(){
        changeDistorsionTypeButtonState(BLUESDRIVER_BUTTON_IDX, distorsionTypeBtns);
       }
    );
    RATButton.mouseClicked( function(){
        changeDistorsionTypeButtonState(RAT_BUTTON_IDX, distorsionTypeBtns);
       }
    );
	//This function will format colors and positions of the DOM elements (sliders, button and text)
	formatDOMElements();
}

function draw() {
    background(5,5,255,1);

    //store values in the buffer
	buffer[BUFFER_ONOFF_IDX] = isFXOn;
	buffer[BUFFER_ACTIVE_FX_IDX] = activeFXButton;

	//SEND BUFFER to Bela. Buffer has index 0 (to be read by Bela),
	//contains floats and sends the 'buffer' array.
    Bela.data.sendBuffer(0, 'float', buffer);
}

function formatDOMElements() {

	//Format Buttons and labels
    //ON/OFF button
    onOffButtonsDimension = 50
    onOffButtonsYpos = windowHeight/2 - onOffButtonsDimension/2
    btnDistance = 125

    //fx buttons
    fxButtonsDimension = 100
    fxButtonsYpos = windowHeight/2 - fxButtonsDimension/2
    btnDistance = 125

    // ON/OFF Button
    posX = fxButtonsDimension
	onOffButton.position(posX, onOffButtonsYpos);
	onOffButton.size(onOffButtonsDimension, onOffButtonsDimension);
    setDefaultFxButtonStyle(onOffButton)

    // Clean Button
    posX = fxButtonsDimension + btnDistance
	cleanButton.position(posX, fxButtonsYpos);
	cleanButton.size(fxButtonsDimension, fxButtonsDimension);
    setDefaultFxButtonStyle(cleanButton)
    cleanButton.style('background-color', onColor);


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
	}

//Function that changes buttonState variable and changes button's background-color
function changeDistorsionTypeButtonState(buttonIdx, buttons) {
    activeFXButton = buttonIdx;

    for (let i = 0; i < buttons.length; i++) {
        buttons[i].style('background-color', offColor);
    }
    buttons[buttonIdx].style('background-color', onColor);
}

//Function that changes buttonState variable and changes button's background-color
function changeOnOffButtonState(buttonIdx, button) {
    if (isFXOn) {
        button.style('background-color', offColor);
        isFXOn = 0;
    } else {
        button.style('background-color', onColor);
        isFXOn = 1;
    }
}

//Function that changes buttonState variable and changes button's background-color
function setDefaultFxButtonStyle(button) {
	let col = color(0, 10, 10, 100);
	button.style('font-weight','bolder');
	button.style('border', '2px solid #000000');
	button.style('border-radius', '50%');
    button.style('background-color', col);
	button.style('color', 'white');
}
