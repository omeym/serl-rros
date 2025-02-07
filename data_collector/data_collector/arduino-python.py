######################
import serial
import time
import subprocess
import signal
import sys

# Use the correct serial port based on what you found, either '/dev/ttyACM0' or '/dev/ttyUSB0'
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1)

def send_command(command):  
    arduino.write(command.encode())  # Send the command to the Arduino
    time.sleep(0.05)  # Small delay to ensure the command is sent properly

def start_two_cameras_script():
    # Launch the two_cameras.py script
    subprocess.Popen(["python", "/home/lm-2023/jeon_team_ws/control_kuka/lbr_fri_ws-dev-socket/data_collection/data_collection/two_cameras_live.py"])

def signal_handler(sig, frame):
    print("Signal received, sending 'X' to microcontroller...")
    send_command('X')  # Send the 'X' command when the process is terminated
    sys.exit(0)  # Exit the program

# Register the signal handler for termination signals   
signal.signal(signal.SIGINT, signal_handler)  # Handles Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handles termination signal

# Example usage
while True:
    command = input("Enter command (X: Press Both Buttons, Y: Release Both Buttons and Start two_cameras.py): ").strip().upper()
    
    if command == 'X':
        send_command('X')  # Press both buttonsy

    elif command == 'Y':   
        send_command('Y')  # Release both buttons
        #start_two_cameras_script()  # Start the two_cameras.py script

    else:
        print("Invalid command. Please enter X or Y.")

