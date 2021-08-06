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


    const video = document.createElement("video");
    document.body.appendChild(video);

    video.width = 500; 
    video.height = 375;

    video.srcObject = stream;
    video.play();

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);

    let canvas = document.createElement("canvas");
    canvas.id="canvasOutput";
    document.body.appendChild(canvas);
    const FPS = 22;

    setInterval(() => {
        cap.read(src);

        var type = "image/png"
        var data = canvas.toDataURL(type);
        data = data.replace('data:' + type + ';base64,', '');

        socket.emit('image', data);
    }, 10000/FPS);
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
