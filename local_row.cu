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
#include "local_row.h"

using namespace cv::cuda;
using namespace cv::cuda::device;
using namespace std;

#define MAX_KERNEL_SIZE 63

__constant__ float c_kernel[MAX_KERNEL_SIZE];

__global__ void linearRowFilter(const PtrStepSz<float> src, PtrStep<float> dst, const int anchor,const int width,const int KSIZE)//, const B brd)//,const int KSIZE)
{
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
            const int BLOCK_DIM_X = 64;
            const int BLOCK_DIM_Y = 4;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE = 1;
        #else
            const int BLOCK_DIM_X = 32;
            const int BLOCK_DIM_Y = 4;
            const int PATCH_PER_BLOCK = 4;
            const int HALO_SIZE = 1;
        #endif

/*	int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        if (x>=src.cols || y>=src.rows)return;
		
	dst(y,x)=src(y,x);*/

		
 //       typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type sum_t;

//        __shared__ sum_t smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];
        __shared__ float smem[BLOCK_DIM_Y][(PATCH_PER_BLOCK + 2 * HALO_SIZE) * BLOCK_DIM_X];


        const int y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;

        if (y >= src.rows)
            return;

        const float* src_row = src.ptr(y);

        const int xStart = blockIdx.x * (PATCH_PER_BLOCK * BLOCK_DIM_X) + threadIdx.x;

        if (blockIdx.x > 0)
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
               // smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X]);
		smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = (src_row[xStart - (HALO_SIZE - j) * BLOCK_DIM_X]);

        }
        else
        {
            //Load left halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
//                smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_low(xStart - (HALO_SIZE - j) * BLOCK_DIM_X, src_row));
                int p1= xStart - (HALO_SIZE - j) * BLOCK_DIM_X,last_col=width-1,at_low = abs(p1) % (last_col+1);
		smem[threadIdx.y][threadIdx.x + j * BLOCK_DIM_X] = src_row[at_low];//(brd.at_low(xStart - (HALO_SIZE - j) * BLOCK_DIM_X, src_row));i
		}
 	}

        if (blockIdx.x + 2 < gridDim.x)
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j)
//                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart + j * BLOCK_DIM_X]);
                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = (src_row[xStart + j * BLOCK_DIM_X]);


            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j)
//                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X]);
                smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] =(src_row[xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X]);

        }
        else
        {
            //Load main data
            #pragma unroll
            for (int j = 0; j < PATCH_PER_BLOCK; ++j){
//                smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + j * BLOCK_DIM_X, src_row));
		int p1=xStart + j * BLOCK_DIM_X , last_col=width-1, at_high = abs(last_col-abs(last_col-p1))%(last_col+1);
//               smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = (brd.at_high(xStart + j * BLOCK_DIM_X, src_row));
               smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X] = src_row[at_high];//(brd.at_high(xStart + j * BLOCK_DIM_X, src_row));
		}
            //Load right halo
            #pragma unroll
            for (int j = 0; j < HALO_SIZE; ++j){
//            smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = saturate_cast<sum_t>(brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row));
//            smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = (brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row));

		int p1=xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X,last_col=width -1, at_high=abs(last_col-abs(last_col-p1))%(last_col+1);

		 smem[threadIdx.y][threadIdx.x + (PATCH_PER_BLOCK + HALO_SIZE) * BLOCK_DIM_X + j * BLOCK_DIM_X] = src_row[at_high]; //(brd.at_high(xStart + (PATCH_PER_BLOCK + j) * BLOCK_DIM_X, src_row));

		}
        }

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PATCH_PER_BLOCK; ++j)
        {
            const int x = xStart + j * BLOCK_DIM_X;

            if (x < src.cols)
            {
//                sum_t sum = VecTraits<sum_t>::all(0);
                float sum = 0;


                #pragma unroll
                for (int k = 0; k < KSIZE; ++k)
                    sum = sum + smem[threadIdx.y][threadIdx.x + HALO_SIZE * BLOCK_DIM_X + j * BLOCK_DIM_X - anchor + k] * c_kernel[k];

//                dst(y, x) = saturate_cast<D>(sum);
                dst(y, x) = sum;//src(y,x);//(sum);

            }
        }
    }

    //template <int KSIZE, typename T, typename D, template<typename> class B>
//    template <int KSIZE, typename T, typename D,template<typename> class B >
void caller_row(cv::cuda::PtrStepSz<float> src, cv::cuda::PtrStepSz<float> dst, const float *kernel,const int anchor, const int cc,const int ksize)
    {
        int BLOCK_DIM_X;
        int BLOCK_DIM_Y;
        int PATCH_PER_BLOCK;

        if (cc >= 20)
        {
            BLOCK_DIM_X = 64;
            BLOCK_DIM_Y = 4;
            PATCH_PER_BLOCK = 4;
        }
        else
        {
            BLOCK_DIM_X = 32;
            BLOCK_DIM_Y = 4;
            PATCH_PER_BLOCK = 4;
        }

        const dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
        const dim3 grid(divUp(src.cols, BLOCK_DIM_X * PATCH_PER_BLOCK), divUp(src.rows, BLOCK_DIM_Y));

cout<<"\nk data in caller:\n";

	for(int i=0;i<ksize*2+1;i++)
		cout<<"\t"<<kernel[i];

		
	cudaMemcpyToSymbol(c_kernel, kernel, (2*ksize+1) * sizeof(float), 0, cudaMemcpyHostToDevice);

	if(anchor==-1)
               linearRowFilter<<<grid, block>>>(src, dst, (ksize*2+1)/2, src.cols,ksize*2+1);
	else
	       linearRowFilter<<<grid, block>>>(src, dst, anchor, src.cols,ksize*2+1);


/*        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
*/
    }
//}