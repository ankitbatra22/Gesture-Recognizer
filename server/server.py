import asyncio
from dataclasses import dataclass
import json
import os

import cv2
from aiohttp import web
from aiohttp_middlewares import cors_middleware

from aiortc import RTCPeerConnection, RTCSessionDescription

import sys
#sys.path.insert(0, r'C:\Users\adity\Desktop\Gesture_Recognizer\Gesture_Recognizer_Model')
user_path = os.getcwd()
path_required = os.path.join(user_path.split('/server')[0], "Gesture_Recognizer_Model")
#print(path_required)
sys.path.insert(0, path_required)

from detector import Detector

pcs = set()
dataChannel = None


detector = Detector()


async def consume_video(track):
    while True:
        frame = await track.recv()
        img = frame.to_ndarray(format="bgr24")

        detector.add_frame(img)
        detector.predict_on_frames()

        cv2.imshow("window", img)
        cv2.waitKey(1)
        #dataChannel.send(json.dumps({"command": "click"}))


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        global dataChannel
        dataChannel = channel

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(consume_video(track))

        @track.on("ended")
        async def on_ended():
            print("stream ended")

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    app = web.Application(middlewares=[cors_middleware(allow_all=True)])
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host="0.0.0.0", port=8080, ssl_context=None
    )
