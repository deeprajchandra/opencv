#include "opencv2/core/cuda/common.hpp"
//#include "opencv2/core/cuda/saturate_cast.hpp"
//#include "opencv2/core/cuda/vec_math.hpp"
//#include "opencv2/core/cuda/border_interpolate.hpp"
#include <limits>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/imgproc.hpp"
//#include "local_border_interpolate.hpp"
//#include "local_row_filter.cu"
//#include "opencv2/core/private.cuda.hpp"
#include <iostream>
#include "local_column.h"

using namespace cv::cuda;
using namespace cv::cuda::device;
using namespace std;

#define MAX_KERNEL_SIZE 63

__constant__ float c_kernel[MAX_KERNEL_SIZE];


__global__ void linearColumnFilter(const PtrStepSz<float> src, PtrStep<float> dst, const int anchor, const int height,const int KSIZE )
    {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            const int BLOCK_DIM_X = 4;
            const int BLOCK_DIM_Y = 64;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE =2;//<= 16 ? 1 : 2;
        #else
            const int BLOCK_DIM_X = 16;
            const int BLOCK_DIM_Y = 8;
            const int PATCH_PER_BLOCK = 2;
            const int HALO_SIZE = 2;
        #endif


        __shared__ float smem[(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_Y][BLOCK_DIM_X];

        const int x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

        if (x >= src.cols)
            return;

        const float* src_col = src.ptr() + x;

        const int yStart = blockIdx.y * (BLOCK_DIM_Y * PATCH_PER_BLOCK) + threadIdx.y;

        if (blockIdx.y > 0)
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = (src(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, x));
        }
        else
        {
            //Upper halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
		int p1= yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, idx_row_low=abs(p1)%height;
                smem[threadIdx.y + j * BLOCK_DIM_Y][threadIdx.x] = (*(const float*)((const char*)src_col + idx_row_low * src.step));
		//(brd.at_low(yStart - (HALO_SIZE - j) * BLOCK_DIM_Y, src_col, src.step));
	    }
        }

        if (blockIdx.y + 2 < gridDim.y)
        {
            //Main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = (src(yStart + j * BLOCK_DIM_Y, x));

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] =(src(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, x));
        }
        else
        {
            //Main data
		#pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j){
		int p1=yStart + j * BLOCK_DIM_Y, idx_row_high=abs((height-1) - abs((height-1) - p1)) % height;
                smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = (*(const float*)((const char*)src_col + idx_row_high * src.step));
			//(brd.at_high(yStart + j * BLOCK_DIM_Y, src_col, src.step));
	    }

            //Lower halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
		int p1=yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, idx_row_high=abs((height-1) - abs((height-1) - p1)) % height;
                smem[threadIdx.y + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_Y + j * BLOCK_DIM_Y][threadIdx.x] = (*(const float*)((const char*)src_col + idx_row_high * src.step));
			//saturate_cast<sum_t>(brd.at_high(yStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_Y, src_col, src.step));
	    }
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int y = yStart + j * BLOCK_DIM_Y;

            if (y < src.rows)
            {
//                sum_t sum = VecTraits<sum_t>::all(0);
		  float sum=0;

                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum = sum + smem[threadIdx.y + HALO_SIZE * BLOCK_DIM_Y + j * BLOCK_DIM_Y - anchor + k][threadIdx.x] * c_kernel[k];

                dst(y, x) = (sum);
            }
        }
    }
	
    void caller_column(PtrStepSz<float> src, PtrStepSz<float> dst, const float *kernel,const int anchor, const int cc,const int ksize)
    {
        int BLOCK_DIM_X;
        int BLOCK_DIM_Y;
        int PATCH_PER_BLOCK;

        if (cc >= 20)
        {
            BLOCK_DIM_X = 4;
            BLOCK_DIM_Y = 64;
            PATCH_PER_BLOCK = 4;
        }
        else
        {
            BLOCK_DIM_X = 16;
            BLOCK_DIM_Y = 8;
            PATCH_PER_BLOCK = 2;
        }


	cout<<"\nk data in caller:\n";

        for(int i=0;i<ksize*2+1;i++)
                cout<<"\t"<<kernel[i];


        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, BLOCK_DIM_X), divUp(src.rows, BLOCK_DIM_Y * PATCH_PER_BLOCK));
	
        cudaMemcpyToSymbol(c_kernel, kernel, (2*ksize+1) * sizeof(float), 0, cudaMemcpyHostToDevice);
	if(anchor==-1)
	        linearColumnFilter<<<grid, block>>>(src, dst, (ksize*2+1)/2,src.rows,ksize*2+1);
	else
                linearColumnFilter<<<grid, block>>>(src, dst, anchor,src.rows,ksize*2+1);



   }
