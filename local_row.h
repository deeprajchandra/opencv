
//void caller_row(PtrStepSz , PtrStepSz , int , int );//, cudaStream_t stream)

void caller_row(cv::cuda::PtrStepSz<float> , cv::cuda::PtrStepSz<float> , const float *,const int , const int,const int);


