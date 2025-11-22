#include <assert.h>

#include "common.h"
#include "timer.h"

#define coarse_factor 2

__global__ void kernel2_nw(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    // __shared__ int Buffer[3][SEQUENCE_LENGTH];
    __shared__ int Buffer1[SEQUENCE_LENGTH];
    __shared__ int Buffer2[SEQUENCE_LENGTH];
    __shared__ int Buffer3[SEQUENCE_LENGTH];
    int* b1 = Buffer1;
    int* b2 = Buffer2;
    int* b3 = Buffer3;

    for (unsigned int i=0; i< 2*SEQUENCE_LENGTH-1; ++i) {
        // int bound = 1 + (i*coarse_factor)/SEQUENCE_LENGTH - (i/(2*SEQUENCE_LENGTH-2-coarse_factor))*((i*coarse_factor)/SEQUENCE_LENGTH);
        for (unsigned int k=0; k < coarse_factor; ++k){

            int row = threadIdx.x+ k* blockDim.x;

            int col = i - (threadIdx.x +k * blockDim.x);

            if ( col >= 0 && col < SEQUENCE_LENGTH ) {
                
                int top = (row == 0)?((col + 1)*DELETION):(b2[row -1]);
                int left = (col == 0)?((row + 1)*INSERTION):(b2[row]);
                int topleft = (row == 0)?(col*DELETION):(col == 0)?(row*INSERTION):(b1[row-1]) ;

                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match = topleft + ((sequence2_d[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1_d[blockIdx.x*SEQUENCE_LENGTH + col])?MATCH:MISMATCH);


                int max = (insertion > deletion)?insertion:deletion;
                max = (match > max)?match:max;
                // Buffer[i%3][threadIdx.x] = max;
                b3[row] = max;
            }
        }
        int* tmp = b1;
        b1 = b2;
        b2 = b3;
        b3 = tmp;

        __syncthreads();
    } 
    
    if (threadIdx.x == 0 ) {
        scores_d[blockIdx.x] = b2[SEQUENCE_LENGTH - 1];
    }
}

void nw_gpu2(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {
    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024
    const unsigned int numThreadsPerBlock= (SEQUENCE_LENGTH+ coarse_factor -1) / coarse_factor;
    const unsigned int numBlocks=numSequences;
    kernel2_nw <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}


