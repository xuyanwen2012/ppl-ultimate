# Justfile

# Set the executable name and paths
executable := "mini-demo"
release_path := "./build/android/armeabi-v7a/release/$(executable)"
debug_path := "./build/android/armeabi-v7a/debug/$(executable)"

# # Rule for specifying which build to use (release or debug)
# set-build target:
# 	if [[ "{{target}}" == "release" ]]; then
# 		local_executable_path := "{{release_path}}"
# 	else
# 		local_executable_path := "{{debug_path}}"
# 	fi

# Rule to list connected devices
list-devices:
	@adb devices | awk 'NR>1 {print $1}' | grep -v '^$' || echo "No devices connected."

# Rule to push and run executable on connected devices
run target args="":
	just set-build "{{target}}"
	@device_list=$(adb devices | awk 'NR>1 {print $1}' | grep -v '^$') || { echo "No devices connected."; exit 1; }
	@for device_serial in $device_list; do \
		echo "Running on device: $$device_serial"; \
		adb -s $$device_serial push ${release_path} /data/local/tmp; \
		adb -s $$device_serial shell "cd /data/local/tmp && ./$(executable) $(args)"; \
		echo "Finished running on device: $$device_serial"; \
		echo "------------------------------------"; \
	done

# Example usage:
# just run release -- args
