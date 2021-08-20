console.log("acquireCamera injected");
let predictionStream = null;
// data channel
let dc = null, dcInterval = null;
let pc = null;
let time_start = null;
pc = createPeerConnection();
checkPermissions();
createDataChannel();

function current_stamp() {
    if (time_start === null) {
        time_start = new Date().getTime();
        return 0;
    } else {
        return new Date().getTime() - time_start;
    }
}

function createDataChannel() {
    dc = pc.createDataChannel('chat', {ordered: true});
    dc.onclose = function() {
        clearInterval(dcInterval);
    };
    dc.onopen = function() {
        dcInterval = setInterval(function() {
            var message = 'ping ' + current_stamp();
            dc.send(message);
        }, 1000);
    };
    dc.onmessage = function(evt) {
        if (evt.data === 'pong') {
            console.log("pong");
        }
        if (evt.data.startsWith("{")) {
            let jsonData = JSON.parse(evt.data);
            if ("command" in jsonData) {
                window.parent.postMessage(evt.data, "*");
            }
        }
    };    
}

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', function(evt) {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
    });

    return pc;
}

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

function endPeerConnection() {
    // close data channel
    if (dc) {
        dc.close();
    }
    
    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function(transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function(sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);    
}

function negotiate() {
    return pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
    }).then(function() {
        // wait for ICE gathering to complete
        return new Promise(function(resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function() {
        var offer = pc.localDescription;
        return fetch('http://localhost:8080/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function(response) {
        return response.json();
    }).then(function(answer) {
        return pc.setRemoteDescription(answer);
    }).catch(function(e) {
        alert(e);
    });
}

function startPredicting(stream) {
    stream.getTracks().forEach(function(track) {
        pc.addTrack(track, stream);
    });
    return negotiate();
}

window.addEventListener("message", (event) => {
    if (event.data == "camera-status-request") {
        checkPermissions();
    } else if (event.data == "start-webcam") {
        console.log("need to start camera");
        navigator.mediaDevices.getUserMedia({video: true}).then(stream => {
            startPredicting(stream);
            predictionStream = stream;
            console.log("successful camera start");
            window.parent.postMessage({message: "init-webcam-status", status: true}, "*");
        }).catch(err => {
            console.error(err);
            window.parent.postMessage({message: "init-webcam-status", status: false}, "*");
        });
    } else if (event.data == "stop-webcam") {
        console.log("need to stop the camera");
        predictionStream.getVideoTracks().forEach(track => track.stop());
        endPeerConnection();
    }
});
