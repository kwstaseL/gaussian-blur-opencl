#define KERNEL_RADIUS 8
__kernel void blur(__global unsigned char* input, __global unsigned char* output, __global float* weights, int width, int height, int axis) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int pixel = y * width + x;
    for (int channel = 0; channel < 4; channel++) { 
        float sum_weight = 0.0f;
        float ret = 0.f;
        for (int offset = -KERNEL_RADIUS; offset <= KERNEL_RADIUS; offset++) {
            int offset_x = axis == 0 ? offset : 0;
            int offset_y = axis == 1 ? offset : 0;
            int pixel_y = clamp(y + offset_y, 0, height - 1);
            int pixel_x = clamp(x + offset_x, 0, width - 1);
            int pixel_index = pixel_y * width + pixel_x;
            float weight = weights[offset + KERNEL_RADIUS];
            ret += weight * input[4 * pixel_index + channel];
            sum_weight += weight;
        }
        output[4 * pixel + channel] = (unsigned char)clamp(ret / sum_weight, 0.f, 255.f);
    }
}
