let cameraPermissionStatus = null;
let cameraOn = false;
let stream = null;

chrome.runtime.onMessage.addListener(
    function (req, sender, resp) {
        switch (req.message) {
            case "camera-status":
                cameraPermissionStatus = (req.status == "approved") ? true : false;
                resp({});
                break;
            case "popup-camera-query":
                if (cameraPermissionStatus === null) {
                    chrome.runtime.sendMessage({message: "bkg-camera-status"}, () => {});
                    resp({message: "camera-status-retry"});
                } else {
                    resp({message: "camera-status", status: cameraPermissionStatus, cameraOn});
                }
                return true;
        }
    }
);

function startPredicting() {
    console.log("starting predictions");
}
