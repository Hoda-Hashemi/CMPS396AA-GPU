#include <assert.h>

#include "common.h"
#include "timer.h"


__global__ void kernel_nw(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences, int* matrix) {
    
    for (unsigned int i=0; i< 2*SEQUENCE_LENGTH-1; ++i) {
        int row = threadIdx.x;
        int col = i - threadIdx.x;
        if ( col >= 0 && col < SEQUENCE_LENGTH ) {
            int top = (row == 0)?((col + 1)*DELETION):(matrix[blockIdx.x*SEQUENCE_LENGTH*SEQUENCE_LENGTH + (row-1)*SEQUENCE_LENGTH + col]);
            int left = (col == 0)?((row + 1)*INSERTION):(matrix[blockIdx.x*SEQUENCE_LENGTH*SEQUENCE_LENGTH + row*SEQUENCE_LENGTH + col - 1]);
            int topleft = (row == 0)?(col*DELETION):((col == 0)?(row*INSERTION):(matrix[blockIdx.x*SEQUENCE_LENGTH*SEQUENCE_LENGTH + (row-1)*SEQUENCE_LENGTH+col-1]));

            int insertion = top + INSERTION;
            int deletion  = left + DELETION;
            int match = topleft + ((sequence2_d[blockIdx.x*SEQUENCE_LENGTH + row] == sequence1_d[blockIdx.x*SEQUENCE_LENGTH + col])?MATCH:MISMATCH);

            int max = (insertion > deletion)?insertion:deletion;
            max = (match > max)?match:max;
            matrix[blockIdx.x*SEQUENCE_LENGTH*SEQUENCE_LENGTH + row*SEQUENCE_LENGTH + col] = max;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0 ) {
        scores_d[blockIdx.x] = matrix[(SEQUENCE_LENGTH*SEQUENCE_LENGTH)*(blockIdx.x + 1) - 1];
    }

}


void nw_gpu0(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences, int* matrix) {

    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024
    const unsigned int numThreadsPerBlock=SEQUENCE_LENGTH;
    const unsigned int numBlocks=numSequences;

    kernel_nw <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences, matrix);

}