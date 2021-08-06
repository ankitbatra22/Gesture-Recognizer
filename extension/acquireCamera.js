let socket;

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
console.log("acquireCamera injected");

function startPredicting(stream) {
    socket = io('http://localhost:5000');

    socket.on('connect', () => {
        console.log("successfully connected to flask server");
    });


    
}

window.addEventListener("message", (event) => {
    if (event.data == "camera-status-request") {
        checkPermissions();
    } else if (event.data == "start-webcam") {
        console.log("need to start camera");
        navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
            startPredicting(stream);
            console.log("successful camera start");
            window.parent.postMessage({message: "init-webcam-status", status: true}, "*");
        }).catch(err => {
            console.error(err);
            window.parent.postMessage({message: "init-webcam-status", status: false}, "*");
        });
    }
});
