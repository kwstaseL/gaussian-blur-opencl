#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cmath>

#define CL_API_SUFFIX__VERSION_2_0
#define CL_EXT_PREFIX__VERSION_2_0_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED

#include "3rdParty/opencl/include/CL/cl.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define KERNEL_RADIUS 8
#define SIGMA 3.0f


bool saveKernelSource(const char* kernel_source, const char* filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    file << kernel_source;
    file.close();
    return true;
}

std::vector<float> precalculateBlurWeights() {
    std::vector<float> weights = std::vector<float>(2 * KERNEL_RADIUS + 1);
    for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; ++offset) {
        weights[offset + KERNEL_RADIUS] = std::exp(-(offset * offset) / (2.0f * SIGMA * SIGMA));
    }
    return weights;
}

int main() {

    const char* filename = "street_night.jpg";
    int width = 0, height = 0, img_orig_channels = 4;

    // loading the image using stb_image
    unsigned char* img_in = stbi_load(filename, &width, &height, &img_orig_channels, 4);
    if (!img_in) {
        std::cout << "Could not load " << filename << "\n";
        return -1;
    }

    // allocating memory for the output image and intermediate buffer
    std::vector<unsigned char> img_out(width * height * 4);
    std::vector<unsigned char> img_intermediate(width * height * 4);

    std::chrono::high_resolution_clock clock;
    auto start = std::chrono::high_resolution_clock::now();

    // platform is essentially a container of devices, we are creating a context for a device
    std::cout << "Creating context" << "\n";
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

    // getting the device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (!device) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
    }
    std::cout << "Device created" << "\n";

    cl_device_type device_type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
    if (device_type == CL_DEVICE_TYPE_GPU) {
        std::cout << "Device type: GPU" << "\n";
    } else if (device_type == CL_DEVICE_TYPE_CPU) {
        std::cout << "Device type: CPU" << "\n";
    }

    // finding the device name
    const int SIZE_DEVICE_NAME = 128;
    char device_name[SIZE_DEVICE_NAME];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(char) * SIZE_DEVICE_NAME, device_name, nullptr);
    std::cout << "Device name: " << device_name << "\n";

    // creating context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    std::cout << "Context created" << "\n";

    // source of kernel code that will be executed on the context device.
    const char* kernel_source =
            "#define KERNEL_RADIUS 8\n"
            "__kernel void blur(__global unsigned char* input, __global unsigned char* output, __global float* weights, int width, int height, int axis) {\n"
            "    int x = get_global_id(0);\n"
            "    int y = get_global_id(1);\n"
            "    int pixel = y * width + x;\n"
            "    for (int channel = 0; channel < 4; channel++) { \n"
            "        float sum_weight = 0.0f;\n"
            "        float ret = 0.f;\n"
            "        for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++) {\n"
            "            int offset_x = axis == 0 ? offset : 0;\n"
            "            int offset_y = axis == 1 ? offset : 0;\n"
            "            int pixel_y = clamp(y + offset_y, 0, height - 1);\n"
            "            int pixel_x = clamp(x + offset_x, 0, width - 1);\n"
            "            int pixel_index = pixel_y * width + pixel_x;\n"
            "            float weight = weights[offset + KERNEL_RADIUS];\n"
            "            ret += weight * input[4 * pixel_index + channel];\n"
            "            sum_weight += weight;\n"
            "        }\n"
            "        output[4 * pixel + channel] = (unsigned char)clamp(ret / sum_weight, 0.f, 255.f);\n"
            "    }\n"
            "}\n";


    // saving the kernel source to a file
    bool success = saveKernelSource(kernel_source, "kernel.cl");
    if (!success) {
        std::cout << "Failed to save the kernel source" << "\n";
        return -1;
    }
    std::cout << "Kernel source saved" << "\n";

    // pre-calculating the weights
    std::vector<float> weights = precalculateBlurWeights();

    // creating a command queue to send commands to the device
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, nullptr);

    // creating an OpenCL buffer to store the weights, input image, intermediate image and output image
    cl_mem weights_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, (2 * KERNEL_RADIUS + 1) * sizeof(float), nullptr, nullptr);
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, width * height * 4 * sizeof(unsigned char), nullptr, nullptr);
    cl_mem intermediate_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * 4 * sizeof(unsigned char), nullptr, nullptr);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * 4 * sizeof(unsigned char), nullptr, nullptr);

    // copying the input image to the input buffer, using the command queue
    cl_int clStatus = clEnqueueWriteBuffer(command_queue, input_buffer, CL_TRUE, 0,
                                           width * height * 4 * sizeof(unsigned char),
                                           img_in, 0, nullptr, nullptr);
    if (clStatus != CL_SUCCESS) {
        std::cout << "Error copying the input image to the input buffer" << "\n";
        return -1;
    }
    clStatus = clEnqueueWriteBuffer(command_queue, weights_buffer, CL_TRUE, 0,
                                    (2 * KERNEL_RADIUS + 1) * sizeof(float),
                                    weights.data(), 0, nullptr, nullptr);

    if (clStatus != CL_SUCCESS) {
        std::cout << "Error copying the weights to the weights buffer" << "\n";
        return -1;
    }

    // creating a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, nullptr, nullptr);

    // build the program
    clStatus = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (clStatus != CL_SUCCESS) {
        std::cout << "Error building program" << "\n";
        return -1;
    }

    // create the kernel
    cl_kernel kernel = clCreateKernel(program, "blur", nullptr);

    // setting the kernel arguments for the horizontal pass
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &intermediate_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    clSetKernelArg(kernel, 4, sizeof(int), &height);

    int axis = 0; // horizontal
    clSetKernelArg(kernel, 5, sizeof(int), &axis);

    // setting the global and local work sizes
    size_t global_work_size[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    size_t local_work_size[2] = {8, 4};

    // executes the kernel for the horizontal pass
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (clStatus != CL_SUCCESS) {
        std::cout << "Error executing kernel for horizontal pass" << "\n";
    }
    clFinish(command_queue);

    // setting the kernel arguments for the vertical pass
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &intermediate_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights_buffer);
    clSetKernelArg(kernel, 3, sizeof(int), &width);
    clSetKernelArg(kernel, 4, sizeof(int), &height);

    axis = 1; // vertical
    clSetKernelArg(kernel, 5, sizeof(int), &axis);

    // execute the kernel for the vertical pass
    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
    if (clStatus != CL_SUCCESS) {
        std::cout << "Error executing kernel for vertical pass" << "\n";
        return -1;
    }
    clFinish(command_queue);

    // copying the output image from the output buffer
    clStatus = clEnqueueReadBuffer(command_queue, output_buffer, CL_TRUE, 0,
                                   width * height * 4 * sizeof(unsigned char),
                                   img_out.data(), 0, nullptr, nullptr);

    if (clStatus != CL_SUCCESS) {
        std::cout << "Error copying the output image from the output buffer" << "\n";
        return -1;
    }

    // write the output image to a file
    bool success_write = stbi_write_jpg("image_blurred_final.jpg", width, height, 4, img_out.data(), 90);
    if (!success_write) {
        std::cout << "Failed to write the output image" << "\n";
        return -1;
    }
    // Timer end
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_diff = end - start;
    std::cout << "Time taken: " << time_diff.count() << " milliseconds" << "\n";

    // Clean up
    clReleaseContext(context);
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(intermediate_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(weights_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    stbi_image_free(img_in);

    return 0;
}
