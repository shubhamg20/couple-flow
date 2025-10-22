import numpy as np
import torch

# The index to map the OpenXR hand joints to the hand joints used
# in Dex-retargeting.
_HAND_JOINTS_INDEX = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25
]

# The transformation matrices to convert hand pose to canonical view.
_OPERATOR2MANO_RIGHT = np.asarray([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])

_OPERATOR2MANO_LEFT = np.asarray([
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
])

import socket
import json
import time


def send_command_to_vision_pro(ip_address: str, port: int, command: str):

    data = command
    json_data = json.dumps(data).encode("utf-8")

    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # s.connect((ip_address, port))
        # data = {"type": "command", "command": command}
        # json_data = json.dumps(data).encode("utf-8")
        # s.sendall(json_data)
        # print(json_data)
        # print(f"✅ Sent command to Vision Pro: {command}")


class VisionProClient:

    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port
        self.last_command_time = time.time()

    def send_command(self, command: str):
        if time.time() - self.last_command_time < 2.0:
            return

        send_command_to_vision_pro(self.ip_address, self.port, command)
        self.last_command_time = time.time()
