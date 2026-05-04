from __future__ import annotations

import time
from threading import Lock

import serial
from flask import Flask
from serial import SerialException


app = Flask(__name__)

PORT = 'COM17'
BAUD = 9600
WRITE_TIMEOUT_SECONDS = 1

arduino: serial.Serial | None = None
last_command: str | None = None
serial_lock = Lock()


def connect_arduino() -> bool:
    global arduino, last_command

    try:
        if arduino is not None and arduino.is_open:
            return True

        arduino = serial.Serial(
            PORT,
            BAUD,
            timeout=1,
            write_timeout=WRITE_TIMEOUT_SECONDS,
            rtscts=False,
            dsrdtr=False,
        )
        arduino.setDTR(False)
        arduino.setRTS(False)
        time.sleep(2)
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        last_command = None
        print(f"Connected to Arduino on {PORT}")
        return True
    except SerialException as error:
        print("Connection error:", error)
        arduino = None
        return False


def close_arduino() -> None:
    global arduino, last_command

    try:
        if arduino is not None and arduino.is_open:
            arduino.close()
    except SerialException as error:
        print("Close error:", error)
    finally:
        arduino = None
        last_command = None


def write_to_arduino(command: str) -> None:
    if arduino is None or not arduino.is_open:
        if not connect_arduino():
            raise SerialException(f"Arduino not connected on {PORT}")

    arduino.write(command.encode("ascii"))
    arduino.flush()


def send_command(command: str):
    global last_command

    with serial_lock:
        if command == last_command:
            return {"status": "ignored", "command": command, "reason": "same_command"}

        try:
            write_to_arduino(command)
        except SerialException as first_error:
            print("Serial write error:", first_error)
            close_arduino()
            time.sleep(0.5)

            try:
                write_to_arduino(command)
            except SerialException as second_error:
                print("Serial retry error:", second_error)
                close_arduino()
                return {
                    "error": "Arduino write failed",
                    "detail": str(second_error),
                    "port": PORT,
                }, 503

        last_command = command
        return {"status": "sent", "command": command}


connect_arduino()


@app.route('/unlock', methods=['GET', 'POST'])
def unlock():
    print("Unlock request")
    return send_command('1')


@app.route('/lock', methods=['GET', 'POST'])
def lock():
    print("Lock request")
    return send_command('0')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
