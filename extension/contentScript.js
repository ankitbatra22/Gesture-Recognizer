let extensionURL = 'chrome-extension://' + chrome.runtime.id;

if (!location.ancestorOrigins.contains(extensionURL)) {
    let iframe = document.createElement('iframe');
    iframe.src = chrome.runtime.getURL('frame.html')
    iframe.style.display = "none";
    iframe.allow = "camera";
    iframe.id = "gesture-recognizer";
    document.body.appendChild(iframe);
}

window.addEventListener("message", (event) => {
    if (event.data == "camera-approved") {
        chrome.runtime.sendMessage({message: "camera-status", status: "approved"}, () => {});
    } else if (event.data == "camera-denied") {
        chrome.runtime.sendMessage({message: "camera-status", status: "denied"}, () => {})
    }
});

chrome.runtime.onMessage.addListener(
    function (req, sender, res) {
        switch (req.message) {
            case "bkg-camera-status":
                document.getElementById("gesture-recognizer").contentWindow.postMessage("camera-status-request");
                res({});
                return true;
            case "start-webcam":
                console.log(navigator);
                res({navigator: navigator});
                return;
        }
    }
);