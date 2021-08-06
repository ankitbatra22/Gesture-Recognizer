let cameraPermission = null;
let cameraOn = false;
let webcamStream = null;

checkCameraPermissions();

function checkCameraPermissions() {
    console.log("checking perms");
    chrome.runtime.sendMessage({message: "popup-camera-query"}, (response) => {
        console.log(response);
        if (response.message == "camera-status") {
            cameraPermission = response.status;
            cameraOn = response.cameraOn;
        } else if (response.message == "camera-status-retry") {
            setTimeout(checkCameraPermissions, 200);
        }
    });
}

function initializeCamera() {
    if (cameraPermission === null) {
        console.error("Popup doesn't know camera permissions yet.");
        //checkCameraPermissions();
        return;
    }

    if (cameraPermission) {
        chrome.runtime.sendMessage({message: "popup-camera-initiate", navigator: JSON.stringify(navigator)}, (response) => {
            if (response.status == "success") {
                navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
                    webcamStream = stream;
                    let vidObj = document.getElementById("webcam-video");
                    vidObj.srcObject = webcamStream;
                }).catch(err => {throw(err);});
                cameraOn = true;
                document.getElementById("cam-start-btn").style.visibility = "hidden";
                document.getElementById("cam-stop-btn").style.visibility = "visible";
            } else {
                alert("There was an error turning on the webcam");
                cameraOn = false;
            }
        });
    } else {
        alert("Gesture Recognizer does not have permission to turn on the camera!");
    }
}

function stopCamera() {
    if (cameraOn) {
        chrome.runtime.sendMessage({message: "popup-camera-stop"}, (response) => {
            webcamStream.getVideoTracks().forEach(track => track.stop());
            document.getElementById("cam-stop-btn").style.visibility = "hidden";
            document.getElementById("cam-start-btn").style.visibility = "visible";
        });
    } else {
        throw("An error has occurred. The camera was not on, and stopCamera was called.");
    }
}

document.getElementById("cam-start-btn").addEventListener("click", initializeCamera);
document.getElementById("cam-stop-btn").addEventListener("click", stopCamera);