let extensionURL = 'chrome-extension://' + chrome.runtime.id;
let cameraOn = false;
let cameraPermission = false;
let innerWindow;

console.log("Content script injected");

if (!location.ancestorOrigins.contains(extensionURL)) {
    let iframe = document.createElement('iframe');
    iframe.src = chrome.runtime.getURL('frame.html')
    iframe.style.display = "none";
    iframe.allow = "camera";
    iframe.id = "gesture-recognizer";
    document.body.appendChild(iframe);
    innerWindow = document.getElementById("gesture-recognizer").contentWindow;
}

window.addEventListener("message", (event) => {
    if (event.data == "camera-approved") {
        chrome.runtime.sendMessage({message: "camera-status", status: "approved"}, () => {});
        cameraPermission = true;
    } else if (event.data == "camera-denied") {
        chrome.runtime.sendMessage({message: "camera-status", status: "denied"}, () => {});
        cameraPermission = false;
    } else if (event.data.message == "init-webcam-status") {
        chrome.runtime.sendMessage({message: "init-webcam-status", status: event.data.status});
        cameraOn = event.data.status;
    }
});

chrome.runtime.onMessage.addListener(
    function (req, sender, res) {
        switch (req.message) {
            case "popup-camera-query":
                console.log("received a query");
                if (cameraPermission === null) {
                    res({message: "camera-status-retry"});
                } else {
                    console.log("sending response");
                    res({message: "camera-status-response", cameraOn, status: cameraPermission});
                }
                return true;
            case "popup-camera-initiate":
                console.log("asking to start the camera");
                innerWindow.postMessage("start-webcam", "*");
                res({});
                break;
        }
    }
);