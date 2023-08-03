// COLORS
// fx button colors button colo
let offColor;
let onColor;
let pedalBackgroundColor;

// BUFFER
// indices
const BUFFER_ONOFF_IDX = 0
const BUFFER_ACTIVE_FX_IDX = 1

// FX BUTTONS
// indices
const CLEAN_BUTTON_IDX = 0
const TUBESCREAMER_BUTTON_IDX = 1
const BLUESDRIVER_BUTTON_IDX = 2
const RAT_BUTTON_IDX = 3

// names
const buttonNames = ['Clean', 'TubeScreamer', 'BluesDriver', 'RAT']

//buffer to send to Bela. 5 elements:
// 4 buttons [clean, TubeScreamer, BluesDriver, RAT]
// 1 slider [Volume]
// buffer: [isFxOn, activeFXButton]
let buffer = [1, 0];
let activeFXButton = 0;
let isFXOn = 0;

function setup() {
	createCanvas(windowWidth, 3300);

    offColor = color(0, 10, 10, 100);
    onColor = color(50, 110, 110, 110);
    pedalBackgroundColor = color(19, 73, 147, 256)

    // Create buttons
	// Create Distorsion type buttons
    onOffButtonCenter = createButton("")
    fxSwitchButtonCenter = createButton("")
    ledIndicator = createDiv()
    cleanFXLedIndicator = createDiv()
    tubeScreamerFXLedIndicator = createDiv()
    bluesDriverFXLedIndicator = createDiv()
    RATFXLedIndicator = createDiv()
    fxLedIndicators = [
        cleanFXLedIndicator,
        tubeScreamerFXLedIndicator,
        bluesDriverFXLedIndicator,
        RATFXLedIndicator
    ]


    activeFXButton = 0;
    attachButtonListeners();
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

function attachButtonListeners() {
	// On/Off Button listener
    onOffButtonCenter.mouseClicked( function(){
        changeOnOffButtonState(isFXOn);
       }
    );

    fxSwitchButtonCenter.mouseClicked( function(){
        selectNextFx();
    })
}

function formatDOMElements() {

    let pedalWrapper = createDiv()
    pedalWrapper.style("position", "relative")
    pedalWrapper.style("left", "-50%")
    pedalWrapper.style("margin-left", "-4.375em")
    pedalWrapper.style("font-family", "Helvetica")

    // ================ PEDAL OUTLINE ================
    let pedalOuter = createDiv()
    pedalOuter.parent(pedalWrapper)
    pedalOuter.style("border-radius", ".5em")
    pedalOuter.style("position", "relative")
    pedalOuter.style("background", "#134993")
    pedalOuter.style("box-shadow", "inset 1px 1px 1px #134993, inset 3px 2px 2px rgba(255,255,255,1), inset -3px -2px 2px rgba(0,0,0,0.25), inset -1px -1px 1px #134993, 6px 8px 10px rgba(0,0,0,0.5)")
    pedalOuter.style("border", "0.2px solid #134993")

    pedalOuter.style('width', '8.75em')
    pedalOuter.style('height', '15.75em')

    let pedalInner = createDiv()
    pedalInner.parent(pedalOuter)
    pedalInner.style("border-radius", ".5em")
    pedalInner.style("position", "relative")
    pedalInner.style("background", "#134993")
    pedalInner.style("box-shadow", "inset 1px 1px 1px #134993, inset 3px 2px 2px rgba(255,255,255,1), inset -3px -2px 2px rgba(0,0,0,0.25), inset -1px -1px 1px #134993, 6px 8px 10px rgba(0,0,0,0.5)")
    pedalInner.style("border", "0.2px solid #134993")

    pedalInner.style("width", "100%")
    pedalInner.style("height", "14.75em")
    pedalInner.style("top", ".5em")
    pedalInner.style("border", "0.2px solid #134993")
    pedalInner.style("box-shadow", " inset 1px 1px 1px #134993, inset 3px 2px 2px rgba(255,255,255,1), inset -3px -2px 2px rgba(0,0,0,0.25)")

    // ================ PEDAL WHITE BORDERS ================
    // --------------------- TOP BORDERS ---------------------
    let borderTop = createDiv()
    borderTop.parent(pedalInner)
    borderTop.style("border", ".4em solid #fffddd")
    borderTop.style("width", "7.3em")
    borderTop.style("height", "8.75em")
    borderTop.style("top", ".5em")
    borderTop.style("left", ".35em")
    borderTop.style("position", "absolute")
    borderTop.style("border-radius", ".25em")
    borderTop.style("border-bottom-right-radius", "0")
    borderTop.style("border-bottom-left-radius", "0")
    borderTop.style("background", "transparent")

    // --------------------- TOP LABELS ---------------------
    let cleanBorderLabel = createSpan("CLEAN")
    cleanBorderLabel.parent(borderTop)
    cleanBorderLabel.style("top", "10%")
    cleanBorderLabel.style("left", "-2.325em")
    cleanBorderLabel.style("transform", " rotate(90deg)")
    setBorderSpanStyle(cleanBorderLabel)

    let tubeScreamerBorderLabel = createSpan("TUBE ")
    tubeScreamerBorderLabel.parent(borderTop)
    tubeScreamerBorderLabel.style("top", "12%")
    tubeScreamerBorderLabel.style("right", "-2.025em")
    tubeScreamerBorderLabel.style("transform", " rotate(-90deg)")
    setBorderSpanStyle(tubeScreamerBorderLabel)

    let bluesDriverBorderLabel = createSpan("BLUES")
    bluesDriverBorderLabel.parent(borderTop)
    bluesDriverBorderLabel.style("top", "45%")
    bluesDriverBorderLabel.style("left", "-2.35em")
    bluesDriverBorderLabel.style("transform", " rotate(90deg)")
    setBorderSpanStyle(bluesDriverBorderLabel)

    let RATBorderLabel = createSpan("RAT")
    RATBorderLabel.parent(borderTop)
    RATBorderLabel.style("top", "45%")
    RATBorderLabel.style("right", "-1.65em")
    RATBorderLabel.style("transform", " rotate(-90deg)")
    setBorderSpanStyle(RATBorderLabel)


    // --------------------- BOT BORDERS ---------------------
    let borderBottom = createDiv()
    borderBottom.parent(pedalInner)
    borderBottom.style("border", ".4em solid #fffddd")
    borderBottom.style("width", "7.3em")
    borderBottom.style("height", "3.1em")
    borderBottom.style("bottom", ".65em")
    borderBottom.style("left", ".35em")
    borderBottom.style("position", "absolute")
    borderBottom.style("background", "transparent")
    borderBottom.style("border-radius", ".25em")
    borderBottom.style("border-top-right-radius", "0")
    borderBottom.style("border-top-left-radius", "0")

    // --------------------- BOT LABELS ---------------------
    let onOffBorderLabel = createSpan("ON / OFF")
    onOffBorderLabel.parent(borderBottom)
    onOffBorderLabel.style("bottom", "-1.1em")
    onOffBorderLabel.style("left", "8%")
    setBorderSpanStyle(onOffBorderLabel)

    let switchFxBorderLabel = createSpan("EFFECT")
    switchFxBorderLabel.parent(borderBottom)
    switchFxBorderLabel.style("bottom", "-1.1em")
    switchFxBorderLabel.style("left", "69%")
    setBorderSpanStyle(switchFxBorderLabel)

    // ================  FX TYPE LEDS ================
    cleanFXLedIndicator.parent(borderTop)
    cleanFXLedIndicator.style("top", "10%")
    cleanFXLedIndicator.style("left", "27.5%")
    setFxTypeLedStyle(cleanFXLedIndicator)
    cleanFXLedIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")

    tubeScreamerFXLedIndicator.parent(borderTop)
    tubeScreamerFXLedIndicator.style("top", "10%")
    tubeScreamerFXLedIndicator.style("left", "72.5%")
    setFxTypeLedStyle(tubeScreamerFXLedIndicator)

    bluesDriverFXLedIndicator.parent(borderTop)
    bluesDriverFXLedIndicator.style("top", "40%")
    bluesDriverFXLedIndicator.style("left", "27.5%")
    setFxTypeLedStyle(bluesDriverFXLedIndicator)

    RATFXLedIndicator.parent(borderTop)
    RATFXLedIndicator.style("top", "40%")
    RATFXLedIndicator.style("left", "72.5%")
    setFxTypeLedStyle(RATFXLedIndicator)


    // ================ PEDAL TITLES ================
    let modelText = createP("FORGER<strong>NFX<strong>")
    modelText.parent(borderTop)
    modelText.style("padding", ".25em")
    modelText.style("font-weight", "100")
    modelText.style("margin", "0")
    modelText.style("bottom", "-0.125em")
    modelText.style("position", "absolute")
    modelText.style("font-size", "1.125em !important")
    modelText.style("letter-spacing", "-2px")
    modelText.style("color", "#fffddd")

    let brandText = createP("S.T.")
    brandText.parent(borderTop)
    brandText.style("position", "absolute")
    brandText.style("font-size", "2.1em")
    brandText.style("font-weight", "1000")
    brandText.style("padding", "0 0")
    brandText.style("bottom", "-.55em")
    brandText.style("color", "#134993")
    brandText.style("text-shadow", "-.025em .025em 0 #fffddd, -.05em .05em 0 #fffddd, -.075em .075em 0 #fffddd, -.1em .1em 0 #fffddd, -.125em .125em 0 #fffddd")
    brandText.style("transform", "scale(.75, .9)")

    // ================ PEDAL BOTTOM SECTION ================
    // ---------------- PEDAL ON/OFF BUTTON ----------------
    let onOffButton = createDiv()
    onOffButton.parent(borderBottom)
    onOffButton.style("position", "absolute")
    onOffButton.style("background", "#888")
    onOffButton.style("height", "2.125em")
    onOffButton.style("width", "2.125em")
    onOffButton.style("border-radius", "50%")
    onOffButton.style("bottom", "50%")
    onOffButton.style("margin-bottom", "-1.067em")
    onOffButton.style("left", "-1.067em")
    onOffButton.style("margin-left", "20%")
    onOffButton.style("box-shadow", "inset 1px 1px 3px rgba(255, 255, 255, 0.5), inset -1px -1px 1px rgba(0, 0, 0, 0.5), 8px 10px 10px rgba(0,0,0,0.5)")

    let onOffButtonHex = createDiv()
    onOffButtonHex.parent(onOffButton)
    onOffButtonHex.style("position", "absolute")
    onOffButtonHex.style("top", ".2em")
    onOffButtonHex.style("left", ".07em")
    onOffButtonHex.style("height", "1.725em")
    onOffButtonHex.style("width", "1.985em")

    let onOffButtonHex1 = createDiv()
    onOffButtonHex1.parent(onOffButtonHex)
    onOffButtonHex1.style("float", "left")
    onOffButtonHex1.style("border-right", ".5em solid #9e9e9e")
    onOffButtonHex1.style("border-top", ".866em solid transparent")
    onOffButtonHex1.style("border-bottom", ".866em solid transparent")

    let onOffButtonHex2 = createDiv()
    onOffButtonHex2.parent(onOffButtonHex)
    onOffButtonHex2.style("float", "left")
    onOffButtonHex2.style("width", "1em")
    onOffButtonHex2.style("height", "1.73em")
    onOffButtonHex2.style("background-color", "#9e9e9e")

    let onOffButtonHex3 = createDiv()
    onOffButtonHex3.parent(onOffButtonHex)
    onOffButtonHex3.style("float", "left")
    onOffButtonHex3.style("width", "0")
    onOffButtonHex3.style("border-left", ".5em solid #9e9e9e")
    onOffButtonHex3.style("border-top", ".866em solid transparent")
    onOffButtonHex3.style("border-bottom", ".866em solid transparent")

    onOffButtonCenter.parent(onOffButton)
    onOffButtonCenter.style("position", "absolute")
    onOffButtonCenter.style("width", "1.5em")
    onOffButtonCenter.style("height", "1.5em")
    onOffButtonCenter.style("background", "#afafaf")
    onOffButtonCenter.style("box-shadow", "inset 0 0 25px rgba(222, 222, 222, 1), inset -2px -2px 1px rgba(0, 0, 0, 0.25)")
    onOffButtonCenter.style("border-radius", "50%")
    onOffButtonCenter.style("left", "50%")
    onOffButtonCenter.style("top", "50%")
    onOffButtonCenter.style("border", "0")
    onOffButtonCenter.style("margin-top", "-.75em")
    onOffButtonCenter.style("margin-left", "-.75em")

    // ---------------- PEDAL FX SWITCH BUTTON ----------------
    let fxSwitchButton = createDiv()
    fxSwitchButton.parent(borderBottom)
    fxSwitchButton.style("position", "absolute")
    fxSwitchButton.style("background", "#888")
    fxSwitchButton.style("height", "2.125em")
    fxSwitchButton.style("width", "2.125em")
    fxSwitchButton.style("border-radius", "50%")
    fxSwitchButton.style("bottom", "50%")
    fxSwitchButton.style("margin-bottom", "-1.067em")
    fxSwitchButton.style("left", "-1.067em")
    fxSwitchButton.style("margin-left", "80%")
    fxSwitchButton.style("box-shadow", "inset 1px 1px 3px rgba(255, 255, 255, 0.5), inset -1px -1px 1px rgba(0, 0, 0, 0.5), 8px 10px 10px rgba(0,0,0,0.5)")

    let fxSwitchButtonHex = createDiv()
    fxSwitchButtonHex.parent(fxSwitchButton)
    fxSwitchButtonHex.style("position", "absolute")
    fxSwitchButtonHex.style("top", ".2em")
    fxSwitchButtonHex.style("left", ".07em")
    fxSwitchButtonHex.style("height", "1.725em")
    fxSwitchButtonHex.style("width", "1.985em")

    let fxSwitchButtonHex1 = createDiv()
    fxSwitchButtonHex1.parent(fxSwitchButtonHex)
    fxSwitchButtonHex1.style("float", "left")
    fxSwitchButtonHex1.style("border-right", ".5em solid #9e9e9e")
    fxSwitchButtonHex1.style("border-top", ".866em solid transparent")
    fxSwitchButtonHex1.style("border-bottom", ".866em solid transparent")

    let fxSwitchButtonHex2 = createDiv()
    fxSwitchButtonHex2.parent(fxSwitchButtonHex)
    fxSwitchButtonHex2.style("float", "left")
    fxSwitchButtonHex2.style("width", "1em")
    fxSwitchButtonHex2.style("height", "1.73em")
    fxSwitchButtonHex2.style("background-color", "#9e9e9e")

    let fxSwitchButtonHex3 = createDiv()
    fxSwitchButtonHex3.parent(fxSwitchButtonHex)
    fxSwitchButtonHex3.style("float", "left")
    fxSwitchButtonHex3.style("width", "0")
    fxSwitchButtonHex3.style("border-left", ".5em solid #9e9e9e")
    fxSwitchButtonHex3.style("border-top", ".866em solid transparent")
    fxSwitchButtonHex3.style("border-bottom", ".866em solid transparent")

    fxSwitchButtonCenter.parent(fxSwitchButton)
    fxSwitchButtonCenter.style("position", "absolute")
    fxSwitchButtonCenter.style("width", "1.5em")
    fxSwitchButtonCenter.style("height", "1.5em")
    fxSwitchButtonCenter.style("background", "#afafaf")
    fxSwitchButtonCenter.style("box-shadow", "inset 0 0 25px rgba(222, 222, 222, 1), inset -2px -2px 1px rgba(0, 0, 0, 0.25)")
    fxSwitchButtonCenter.style("border-radius", "50%")
    fxSwitchButtonCenter.style("left", "50%")
    fxSwitchButtonCenter.style("top", "50%")
    fxSwitchButtonCenter.style("border", "0")
    fxSwitchButtonCenter.style("margin-top", "-.75em")
    fxSwitchButtonCenter.style("margin-left", "-.75em")

    // ----------------  ON/OFF LED INDICATOR ----------------
    ledIndicator.parent(borderBottom)
    ledIndicator.style("position", "absolute")
    ledIndicator.style("width", ".45em")
    ledIndicator.style("height", ".45em")
    ledIndicator.style("border", "1px solid #999")
    ledIndicator.style("border-radius", "50%")
    ledIndicator.style("background", "rgba(0, 179, 244, 0.25)")
    ledIndicator.style("background", "radial-gradient(ellipse at center, rgba(255, 355, 255,1) 0%,rgba(32,124,202,1) 49%,rgba(41,137,216,1) 50%, rgba(30,87,153,1) 100%)")
    ledIndicator.style("border-radius", "50%")
    ledIndicator.style("bottom", "50%")
    ledIndicator.style("margin-bottom", "-0.225em")
    ledIndicator.style("left", "50%")
    ledIndicator.style("margin-left", "-0.225em")
    ledIndicator.style("box-shadow", "inset 0 0 25px rgba(0, 0, 0, 0.75)")
}

function setBorderSpanStyle(spanReference){
    spanReference.style("font-weight", "800")
    spanReference.style("color", "#fffddd")
    spanReference.style("font-size", ".5em !important")
    spanReference.style("background", "#134993")
    spanReference.style("letter-spacing", "-0.6px")
    spanReference.style("padding", "0 .25em")
    spanReference.style("margin", "0")
    spanReference.style("position", "absolute")
    spanReference.style("line-height", "1.2em")
    spanReference.style("font-size", "0.4em")
    spanReference.style("font-family", "Helvetica")

}

function setFxTypeLedStyle(ledReference){
    ledReference.style("position", "absolute")
    ledReference.style("width", "1.2em")
    ledReference.style("height", "1.2em")
    ledReference.style("border", "1px solid #999")
    ledReference.style("border-radius", "50%")
    ledReference.style("background", "rgba(0, 179, 244, 0.25)")
    ledReference.style("background", "radial-gradient(ellipse at center, rgba(255, 355, 255,1) 0%,rgba(32,124,202,1) 49%,rgba(41,137,216,1) 50%, rgba(30,87,153,1) 100%)")
    ledReference.style("border-radius", "50%")
    ledReference.style("margin-bottom", "-0.6em")
    ledReference.style("margin-left", "-0.6em")
    ledReference.style("box-shadow", "inset 0 0 25px rgba(0, 0, 0, 0.75)")

}

//Function that changes buttonState variable and changes button's background-color
function changeOnOffButtonState() {
    if (isFXOn) {
        onOffButtonCenter.style('background-color', "#afafaf");
        ledIndicator.style("box-shadow", "inset 0 0 25px rgba(0, 0, 0, 0.75)")
        isFXOn = 0;
    } else {
        onOffButtonCenter.style('background-color', "#999");
        ledIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")
        isFXOn = 1;
        onOffButtonCenter.keyPre
    }
}

// select next FX
function selectNextFx() {
    activeFXButton += 1
    if (activeFXButton>=4){
        activeFXButton = 0
    }

    // disable leds
    for (let index = 0; index < fxLedIndicators.length; index++) {
        fxLedIndicators[index].style("box-shadow", "inset 0 0 25px rgba(0, 0, 0, 0.75)")
    }
    // activate correct led
    switch(activeFXButton){
        case 0:
            cleanFXLedIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")
            break
        case 1:
            tubeScreamerFXLedIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")
            break
        case 2:
            bluesDriverFXLedIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")
            break
        case 3:
            RATFXLedIndicator.style("box-shadow", "0 0 25px rgba(255, 255, 255, 1), 0 0 25px rgba(0, 179, 244, 1)")
            break
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
