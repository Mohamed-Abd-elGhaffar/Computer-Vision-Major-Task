import socket
import struct

# 1. Listen on all available network interfaces
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for Pendulum data on port {UDP_PORT}...")

while True:
    # 2. We are sending 4 'doubles' (32 bytes total)
    data, addr = sock.recvfrom(32)

    # 3. Unpack 4 doubles (d)
    states = struct.unpack('4d', data)

    # Format the data for presentation
    print(f"Arm: {states[0]:.2f} rad | Pendulum: {states[2]:.2f} rad")