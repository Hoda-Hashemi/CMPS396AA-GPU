
#include <assert.h>

#include "common.h"
#include "timer.h"
#define coarse_factor 8

__global__ void kernel3_nw(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {

    __shared__ int Buffer0[SEQUENCE_LENGTH];
    __shared__ int Buffer1[SEQUENCE_LENGTH];
    __shared__ unsigned char sequence1[SEQUENCE_LENGTH];

    int* b0 = Buffer0;
    int* b1 = Buffer1;

    int previous_top[coarse_factor];
    unsigned char my_char[coarse_factor];

    int left;
    int topleft;
    int top;

    #pragma unroll
    for (unsigned int k=0; k<coarse_factor; ++k){
        sequence1[k*blockDim.x + threadIdx.x] = sequence1_d[blockIdx.x*SEQUENCE_LENGTH + k*blockDim.x + threadIdx.x];
        my_char[k] = sequence2_d[blockIdx.x*SEQUENCE_LENGTH + threadIdx.x+k*blockDim.x];
        previous_top[k] = 0;
    }
        
    #pragma unroll 
    for (unsigned int L=0; L < coarse_factor; ++L) {
        for(unsigned int i = 0 ; i < blockDim.x; ++i){
            #pragma unroll
            for (unsigned int k =0 ; k < L+1 ; ++k){
                int row = threadIdx.x+ k* blockDim.x ;
                int col = i - row + L*blockDim.x;
                if ( col >= 0 ) {
                    if(k==0 &&L==0){
                        top = (row == 0)?((col + 1)*DELETION):(b0[row-1]) ;
                        left = (col == 0)?((row + 1)*INSERTION):(b0[row]) ;
                        topleft = (row == 0)?(col*DELETION):(col == 0)?(row*INSERTION):(previous_top[k]);
                    }
                    else if(k==0){
                        top = (row == 0)?((col + 1)*DELETION):(b0[row-1]) ;
                        left = b0[row];
                        topleft = previous_top[k];
                    }
                    else if(k==L){
                        top = b0[row-1] ;
                        left = (col == 0)?((row + 1)*INSERTION):(b0[row]) ;
                        topleft = previous_top[k];
                    }
                    else{
                        top = b0[row-1] ;
                        left = b0[row] ;
                        topleft = previous_top[k];
                    }

                    int insertion = top + INSERTION;
                    int deletion  = left + DELETION;
                    int match = topleft + ((my_char[k] == sequence1[col])?MATCH:MISMATCH);

                    int max = (insertion > deletion)? insertion:deletion;
                    max = (match > max)?match:max;
                    b1[row] = max;
                    previous_top[k] = top;
                } else {break;}
            }
            int* tmp = b0;
            b0 = b1;
            b1 = tmp;
            __syncthreads();
        }
    }
    
    #pragma unroll
    for (unsigned int i=0; i< blockDim.x ; ++i){
        #pragma unroll
        for (unsigned int k =0; k < coarse_factor; ++k){
            int row = threadIdx.x+ k* blockDim.x;
            int col = i - row + SEQUENCE_LENGTH;
            if ( col < SEQUENCE_LENGTH ) {
                int top = (b0[row-1]) ;
                int left = (b0[row]) ;
                int topleft = (previous_top[k]);
                int insertion = top + INSERTION;
                int deletion  = left + DELETION;
                int match = topleft + ((my_char[k] == sequence1[col])?MATCH:MISMATCH);
                int max = (insertion > deletion)? insertion:deletion;
                max = (match > max)?match:max;
                b1[row] = max;
                previous_top[k] = top;
            } else {continue;}
        }
        int* tmp = b0;
        b0 = b1;
        b1 = tmp;
        __syncthreads();   
    }

 
    #pragma unroll
    for (unsigned int L= 1; L < coarse_factor ; ++L) {
        for(unsigned int i=0; i< blockDim.x ; ++i) {
            #pragma unroll
            for (unsigned int k = L  ; k < coarse_factor; ++k){
                int row = threadIdx.x+ k* blockDim.x;
                int col = i - row + SEQUENCE_LENGTH + L*blockDim.x;
                if (col < SEQUENCE_LENGTH ) {
                    int top = (b0[row-1]) ;
                    int left = (b0[row]) ;
                    int topleft = (previous_top[k]);

                    int insertion = top + INSERTION;
                    int deletion  = left + DELETION;
                    int match = topleft + ((my_char[k] == sequence1[col])?MATCH:MISMATCH);

                    int max = (insertion > deletion)? insertion:deletion;
                    max = (match > max)?match:max;
                    b1[row] = max;
                    previous_top[k] = top;
                } else {continue;}
            }
            int* tmp = b0;
            b0 = b1;
            b1 = tmp;
            __syncthreads();
        }
    }
    
    if (threadIdx.x == 0 ) {
        scores_d[blockIdx.x] = b1[SEQUENCE_LENGTH - 1];
    }
}

void nw_gpu3(unsigned char* sequence1_d, unsigned char* sequence2_d, int* scores_d, unsigned int numSequences) {
    assert(SEQUENCE_LENGTH <= 1024); // You can assume the sequence length is not more than 1024
    const unsigned int numThreadsPerBlock=(SEQUENCE_LENGTH+ coarse_factor -1) / coarse_factor;
    const unsigned int numBlocks=numSequences;
    kernel3_nw <<< numBlocks, numThreadsPerBlock >>> (sequence1_d, sequence2_d, scores_d, numSequences);
}
