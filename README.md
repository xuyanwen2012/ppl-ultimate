## Pipeline Build Instructions

### Step 1: Clone the Project
First, clone the `android` branch of the `ppl-ultimate` repository from GitHub:
```bash
git clone -b android https://github.com/xuyanwen2012/ppl-ultimate.git
```
### Configure
```bash
cd ppl-ultimate
xmake f -p android --ndk_sdkver=24
```
### Build
```bash
xmake
```
### Run on Device
To run the project on your Android device, you can execute the script with an optional `--cores` parameter to specify thread pinning to particular CPU cores:
```bash
./run_adb.sh --cores 0 1 2 3 4 ...
```

You can also use the Google Benchmark command line arguments found [here](https://github.com/google/benchmark/blob/main/docs/user_guide.md)

```bash
./run_adb.sh --cores 0 1 2 3 --benchmark_format=<console|json|csv> --benchmark_out=<filename> --benchmark_out_format={json|console|csv} ...
```


## Results
#### Google Pixel 6 - All Cores
```
------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations
------------------------------------------------------------------------------------
CPU/BM_Morton/Threads:1/iterations:50           68.1 ms        0.181 ms           50
CPU/BM_Morton/Threads:2/iterations:50           41.9 ms        0.327 ms           50
CPU/BM_Morton/Threads:4/iterations:50           26.2 ms        0.462 ms           50
CPU/BM_Morton/Threads:8/iterations:50           22.5 ms        0.385 ms           50

CPU/BM_Sort/Threads:1/iterations:50              110 ms        0.552 ms           50
CPU/BM_Sort/Threads:2/iterations:50              131 ms        0.465 ms           50
CPU/BM_Sort/Threads:4/iterations:50              121 ms        0.572 ms           50
CPU/BM_Sort/Threads:8/iterations:50             53.7 ms        0.311 ms           50

CPU/BM_RemoveDup/Threads:1/iterations:50        12.4 ms        0.083 ms           50
CPU/BM_RemoveDup/Threads:2/iterations:50        12.0 ms        0.063 ms           50
CPU/BM_RemoveDup/Threads:4/iterations:50        10.9 ms        0.080 ms           50
CPU/BM_RemoveDup/Threads:8/iterations:50        13.8 ms        0.069 ms           50

CPU/BM_RadixTree/Threads:1/iterations:50         483 ms        0.213 ms           50
CPU/BM_RadixTree/Threads:2/iterations:50         265 ms        0.728 ms           50
CPU/BM_RadixTree/Threads:4/iterations:50         175 ms        0.829 ms           50
CPU/BM_RadixTree/Threads:8/iterations:50         138 ms        0.940 ms           50

CPU/BM_EdgeCount/Threads:1/iterations:50        13.7 ms        0.119 ms           50
CPU/BM_EdgeCount/Threads:2/iterations:50       10.00 ms        0.089 ms           50
CPU/BM_EdgeCount/Threads:4/iterations:50        8.53 ms        0.095 ms           50
CPU/BM_EdgeCount/Threads:8/iterations:50        5.21 ms        0.145 ms           50

CPU/BM_EdgeOffset/Threads:1/iterations:50      0.026 ms        0.011 ms           50
CPU/BM_EdgeOffset/Threads:2/iterations:50      0.022 ms        0.011 ms           50
CPU/BM_EdgeOffset/Threads:4/iterations:50      0.031 ms        0.015 ms           50
CPU/BM_EdgeOffset/Threads:8/iterations:50      0.021 ms        0.012 ms           50

CPU/BM_Octree/Threads:1/iterations:50            145 ms        0.272 ms           50
CPU/BM_Octree/Threads:2/iterations:50           94.4 ms        0.656 ms           50
CPU/BM_Octree/Threads:4/iterations:50           78.7 ms        0.869 ms           50
CPU/BM_Octree/Threads:8/iterations:50           69.8 ms         1.01 ms           50
```
#### Google Pixel 6 - Small Cores
```
------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations
------------------------------------------------------------------------------------
CPU/BM_Morton/Threads:1/iterations:50            297 ms        0.719 ms           50
CPU/BM_Morton/Threads:2/iterations:50            149 ms        0.991 ms           50
CPU/BM_Morton/Threads:4/iterations:50           77.8 ms        0.879 ms           50

CPU/BM_Sort/Threads:1/iterations:50              447 ms         2.95 ms           50
CPU/BM_Sort/Threads:2/iterations:50              228 ms         2.73 ms           50
CPU/BM_Sort/Threads:4/iterations:50              120 ms         3.90 ms           50

CPU/BM_RemoveDup/Threads:1/iterations:50        44.8 ms        0.467 ms           50
CPU/BM_RemoveDup/Threads:2/iterations:50        45.1 ms        0.528 ms           50
CPU/BM_RemoveDup/Threads:4/iterations:50        44.7 ms        0.484 ms           50

CPU/BM_RadixTree/Threads:1/iterations:50        1522 ms        0.517 ms           50
CPU/BM_RadixTree/Threads:2/iterations:50         749 ms        0.693 ms           50
CPU/BM_RadixTree/Threads:4/iterations:50         405 ms         1.01 ms           50

CPU/BM_EdgeCount/Threads:1/iterations:50        68.9 ms        0.604 ms           50
CPU/BM_EdgeCount/Threads:2/iterations:50        34.8 ms        0.764 ms           50
CPU/BM_EdgeCount/Threads:4/iterations:50        18.2 ms        0.911 ms           50

CPU/BM_EdgeOffset/Threads:1/iterations:50      0.070 ms        0.048 ms           50
CPU/BM_EdgeOffset/Threads:2/iterations:50      0.064 ms        0.032 ms           50
CPU/BM_EdgeOffset/Threads:4/iterations:50      0.053 ms        0.027 ms           50

CPU/BM_Octree/Threads:1/iterations:50            748 ms         1.38 ms           50
CPU/BM_Octree/Threads:2/iterations:50            388 ms         1.74 ms           50
CPU/BM_Octree/Threads:4/iterations:50            202 ms         1.71 ms           50
```


#### Google Pixel 6 - Medium Cores
```
------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations
------------------------------------------------------------------------------------
CPU/BM_Morton/Threads:1/iterations:50            119 ms        0.198 ms           50
CPU/BM_Morton/Threads:2/iterations:50           59.4 ms        0.320 ms           50

CPU/BM_Sort/Threads:1/iterations:50              254 ms        0.680 ms           50
CPU/BM_Sort/Threads:2/iterations:50              141 ms         1.13 ms           50

CPU/BM_RemoveDup/Threads:1/iterations:50        14.6 ms        0.134 ms           50
CPU/BM_RemoveDup/Threads:2/iterations:50        15.8 ms        0.103 ms           50

CPU/BM_RadixTree/Threads:1/iterations:50         665 ms        0.209 ms           50
CPU/BM_RadixTree/Threads:2/iterations:50         345 ms        0.351 ms           50

CPU/BM_EdgeCount/Threads:1/iterations:50        22.0 ms        0.150 ms           50
CPU/BM_EdgeCount/Threads:2/iterations:50        10.5 ms        0.201 ms           50

CPU/BM_EdgeOffset/Threads:1/iterations:50      0.067 ms        0.032 ms           50
CPU/BM_EdgeOffset/Threads:2/iterations:50      0.067 ms        0.035 ms           50

CPU/BM_Octree/Threads:1/iterations:50            217 ms        0.359 ms           50
CPU/BM_Octree/Threads:2/iterations:50            130 ms        0.454 ms           50
```

#### Google Pixel 6 - Big Cores
```
------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations
------------------------------------------------------------------------------------
CPU/BM_Morton/Threads:1/iterations:50           64.9 ms        0.163 ms           50
CPU/BM_Morton/Threads:2/iterations:50           34.2 ms        0.184 ms           50

CPU/BM_Sort/Threads:1/iterations:50              108 ms        0.401 ms           50
CPU/BM_Sort/Threads:2/iterations:50             50.9 ms        0.851 ms           50

CPU/BM_RemoveDup/Threads:1/iterations:50        11.3 ms        0.092 ms           50
CPU/BM_RemoveDup/Threads:2/iterations:50        12.4 ms        0.064 ms           50

CPU/BM_RadixTree/Threads:1/iterations:50         508 ms        0.199 ms           50
CPU/BM_RadixTree/Threads:2/iterations:50         267 ms        0.355 ms           50

CPU/BM_EdgeCount/Threads:1/iterations:50        15.1 ms        0.124 ms           50
CPU/BM_EdgeCount/Threads:2/iterations:50        6.48 ms        0.147 ms           50

CPU/BM_EdgeOffset/Threads:1/iterations:50      0.061 ms        0.037 ms           50
CPU/BM_EdgeOffset/Threads:2/iterations:50      0.064 ms        0.035 ms           50

CPU/BM_Octree/Threads:1/iterations:50            141 ms        0.255 ms           50
CPU/BM_Octree/Threads:2/iterations:50           86.8 ms        0.433 ms           50
```







