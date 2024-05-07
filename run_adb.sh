#!/bin/bash

# Executable name
executable="bench-cpu"
# Path to local executable
local_executable_path="./build/android/armeabi-v7a/debug/$executable"

# Push the executable to the Android device
adb push $local_executable_path /data/local/tmp

# Run the executable on the Android device with arguments
adb shell "cd /data/local/tmp && ./$executable $@"
