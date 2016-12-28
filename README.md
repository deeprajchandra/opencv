Produced by:
Developed by: DEEPRAJ CHANDRA
Project Under: Prof. Ravi S. Hegde

SEPARABLE FILTER
In CUDA, the ready made implementation of the separable filter has a disadvantage that it cannot process beyond the kernel size 16. The reason behind that is the kernel configuration for the two sub-filters called row filter and column filter.
Both of these filters are implemented for various types of BORDERS. And both of the sub filter have a unique kernel configuration. 
The row filter works on rows, so the number of threads in X dimension were 32. Whereas, column filter works columns, so the number of threads in the Y dimension were 32. And the MAX_KERNEL_SIZE is also set to 32.
We know that the maximum number of threads in one block is 1024. So, in order to achieve maximum throughput, for row filter the dimension of one block is 32x32. Now by increasing X dimension and reducing the Y dimension but keeping XxY equal to 1024, we can also increase the capability of the separable filter. Inverse goes with the column filter as well. We changed X to 64 and reduced Y to 16. And in column filter we reduced X to 16 and increased Y to 16.
However, we had to pull out code from the opencv and cuda libraries, and due to some issues with saturate_cast.h header file, we faced a lot of issues when compiled the code. As our necessity was to only develop the code for the BORDERREFLECT101, we were able to create that. And the code is giving almost similar time.
Things to remember while using the code
Please note the following details are to be considered:
1.	Anchor Point should be (-1,-1). We developed this code for a specific application.
2.	The border type should be BORDERREFLECT101. We edited the code such that it works for BORDERREFLECT101. If you need for any other border type, kindly specify. I’ll try to make it.
3.	Please see the source and destination type of the images.
4.	Before using the algorithm, convert the images to CV_32F. In the cuda version of the sepFilter, the images are converted to CV_32F. Here in this code, we are not taking care about these things.
5.	Kernel should be passed as a float pointer.
6.	Ksize is not the actual kernel size. If ksize is 16, then the actual kernel size becomes 33.
