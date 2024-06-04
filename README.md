# Gaussian Blur in OpenCL

An example application case of OpenCL demonstrating the use of GPU for parallel programming.

# Timings

| Local Dimensions | First Execution | Second Execution | Third Execution | Fourth Execution | Average  |
| ---------------- | --------------- | ---------------- | --------------- | ---------------- | -------- |
| 32x1             | 0.332 sec       | 0.158 sec        | 0.403 sec       | 0.153 sec        | 0.261 sec |
| 1x32             | 0.371 sec       | 0.250 sec        | 0.197 sec       | 0.157 sec        | 0.243 sec |
| 8x4              | 0.229 sec       | 0.154 sec        | 0.150 sec       | 0.145 sec        | 0.169 sec |
| 4x8              | 0.158 sec       | 0.144 sec        | 0.144 sec       | 0.146 sec        | 0.148 sec |
| 16x2             | 0.164 sec       | 0.150 sec        | 0.144 sec       | 0.155 sec        | 0.153 sec |
| 2x16             | 0.158 sec       | 0.158 sec        | 0.149 sec       | 0.146 sec        | 0.152 sec |
| 32x8             | 0.321 sec       | 0.212 sec        | 0.201 sec       | 0.177 sec        | 0.227 sec |
| 8x32             | 0.191 sec       | 0.188 sec        | 0.160 sec       | 0.156 sec        | 0.173 sec |
| 16x16            | 0.253 sec       | 0.149 sec        | 0.141 sec       | 0.142 sec        | 0.171 sec |
