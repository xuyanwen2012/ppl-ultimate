#!/bin/bash

# Executable name
executable="mini-demo"
# Path to local executable
local_executable_path="./build/android/armeabi-arm64-v8a/release/$executable"
# local_executable_path="./build/android/armeabi-v7a/release/$executable"
# local_executable_path="./build/android/armeabi-v7a/debug/$executable"

# Get the list of all connected device serial numbers
device_list=$(adb devices | awk 'NR>1 {print $1}' | grep -v '^$')

if [ -z "$device_list" ]; then
    echo "No devices connected."
    exit 1
fi

# Iterate over each device and run commands
for device_serial in $device_list; do
    echo "Running on device: $device_serial"
    
    # Push the executable to the Android device
    adb -s $device_serial push $local_executable_path /data/local/tmp
    
    echo ""
    
    # Run the executable on the Android device with arguments
    adb -s $device_serial shell "cd /data/local/tmp && ./$executable $@"
    
    echo "Finished running on device: $device_serial"
    echo "------------------------------------"
done