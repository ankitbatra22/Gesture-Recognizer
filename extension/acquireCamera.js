function askForPermissions() {
    navigator.mediaDevices.getUserMedia({video: true, audio: false})
        .then((stream) => {
            stream.getVideoTracks().forEach(track => track.stop());
            window.parent.postMessage("camera-approved", "*");
        })
        .catch((err) => {
            window.parent.postMessage("camera-denied", "*");
        });
}

function checkPermissions() {
    navigator.permissions.query({name: "camera"}).then((res) => {
        if (res.state == "granted") {
            window.parent.postMessage("camera-approved", "*");
        } else if (res.state == "denied") {
            window.parent.postMessage("camera-denied", "*");
        } else {
            askForPermissions();
        }
    });
}

checkPermissions();

window.addEventListener("message", (event) => {
    if (event.data == "camera-status-request") {
        checkPermissions();
    }
});
