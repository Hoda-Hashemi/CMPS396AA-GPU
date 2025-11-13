#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    __shared__ float A_s[IN_TILE_DIM][IN_TILE_DIM];

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;  
    int inRow = outRow - FILTER_RADIUS + maskRow;
    int inCol = outCol -FILTER_RADIUS +maskCol; 

    for (unsigned int tile=0; tile <(width +IN_TILE_DIM-1)/ IN_TILE_DIM; ++tile){

        // Load input tile into shared memory
        if (inRow< height && tile*IN_TILE_DIM+threadIdx.x < width){
            A_s[threadIdx.y][threadIdx.x] = input[row*width + tile*IN_TILE_DIM + threadIdx.x];}
        else{ A_s[threadIdx.y][threadIdx.x] =0.0f;} 
        __syncthreads();

    if (outRow <height && outCol <width){
        float sum =0.0f;
         for (int maskCol=0; maskCol < MASK_DIM; ++maskCol){

            if(inRow < height && inRow >= 0 && inCol < width && inCol >= 0){
                sum += mask_c[maskRow][maskCol] * input[inRow * width + inCol];
                __syncthreads();
            }
         }
    }
    output[outRow * width + outCol] = sum;
}
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {

    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM*FILTER_DIM*sizeof(float));

}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {

    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 numBlocks((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convolution_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);

}
