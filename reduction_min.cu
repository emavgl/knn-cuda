#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>

#define restrict __restrict__

void check_error(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "%s : errore %d (%s)\n",
      msg, err, cudaGetErrorString(err));
    exit(err);
  }
}

float runtime;
void PrintStats(size_t bytes, cudaEvent_t before, cudaEvent_t after, const char *msg)
{ 
  check_error(cudaEventElapsedTime(&runtime, before, after), msg);
  printf("%s %gms, %g GB/s\n", msg, runtime, bytes/runtime/(1024*1024));
}

__global__ void
init(int * restrict input,  int numels)
{
  int gid = threadIdx.x + blockIdx.x*blockDim.x;
  if (gid < numels)
      input[gid] = gid+1; 
}

 /*shared memory da utilizzare nei medoti di riduzione */
 extern __shared__ int sPartial[];

__global__ void
reduction(int* restrict input, int* restrict output, int numels)
{

  
    int gid = threadIdx.x + blockIdx.x*blockDim.x;
    int min = input[gid];
  
    //Fase 1: pre-riduzione e riempimento shared-memory
    while (gid < numels)
    {
      if(input[gid] < min)
          min = input[gid];
      
      gid += gridDim.x*blockDim.x;
    }
  
    const int lid = threadIdx.x;
    sPartial[lid] = min;
  
    //Fase 2: riduzione in shared memory
    int stride = (blockDim.x)/2;
    while (stride > 0)
    {
      __syncthreads();
      if (lid < stride && sPartial[lid + stride] < sPartial[lid]) {
        sPartial[lid] = sPartial[lid + stride];
      }
      stride /= 2;
    }
  
    /* Fase 3: salvataggio del risultato del blocco in memoria globale */
     if (lid == 0){
        output[blockIdx.x] = sPartial[0];
  }
}

bool check_result(int result, int numels)
{
    return result == 1;
}

int main(int argc, char *argv[])
{
  int numels;

  if (argc > 1) {
    numels = atoi(argv[1]);
  } else {
    fprintf(stderr, "inserire numero di elementi\n");
    exit(1);
  }
  
  int h_output;
  int* d_input, *d_output, *d_result;
  size_t numbytes = numels*sizeof(int);
  
  check_error(cudaMalloc(&d_input, numbytes), "alloc d_input");
  
  cudaEvent_t before_init, before_reduction, before_final_reduction, before_download;
  cudaEvent_t after_init, after_reduction, after_final_reduction, after_download;
  
  check_error(cudaEventCreate(&before_init), "create before_init cudaEvent");
  check_error(cudaEventCreate(&before_reduction), "create before_reduction cudaEvent");
  check_error(cudaEventCreate(&before_final_reduction), "create before_final_reduction cudaEvent");
  check_error(cudaEventCreate(&before_download), "create before_download cudaEvent");
  
  check_error(cudaEventCreate(&after_init), "create after_init cudaEvent");
  check_error(cudaEventCreate(&after_reduction), "create after_reduction cudaEvent");
  check_error(cudaEventCreate(&after_final_reduction), "create after_final_reduction cudaEvent");
  check_error(cudaEventCreate(&after_download), "create after_download cudaEvent");
  
  const int blockSize = 32; //prova a modificare con numeri potenze del 2
  int numBlocks = (numels + blockSize - 1)/blockSize; 
  
  cudaEventRecord(before_init);
  init<<<numBlocks, blockSize>>>(d_input, numels);
  cudaEventRecord(after_init);
  
  check_error(cudaMalloc(&d_output, numBlocks*sizeof(int)), "alloc d_output");
  check_error(cudaMalloc(&d_result, sizeof(int)), "alloc d_result");
  
  cudaEventRecord(before_reduction);
  reduction<<<numBlocks, blockSize, blockSize*sizeof(int)>>>(d_input, d_output, numels);
  cudaEventRecord(after_reduction);
   
  cudaEventRecord(before_final_reduction);
  reduction<<<1, blockSize, blockSize*sizeof(int)>>>(d_output, d_result, numBlocks);
  cudaEventRecord(after_final_reduction);
  
  //copy result from Device to Host (recorded)
  cudaEventRecord(before_download);
  check_error(cudaMemcpy(&h_output, d_result, sizeof(int), cudaMemcpyDeviceToHost), "copy d_result");
  cudaEventRecord(after_download);
  
  //cudaEventElapsedTime
  check_error(cudaEventSynchronize(after_download), "sync cudaEvents");
  PrintStats(numels*sizeof(int), before_init, after_init, "time init");
  PrintStats((numels+numBlocks)*sizeof(int), before_reduction, after_reduction, "time reduction");
  PrintStats((numels+1)*sizeof(int), before_final_reduction, after_final_reduction, "time final_reduction");
  PrintStats(sizeof(int), before_download, after_download, "time download");

  if(!check_result(h_output, numels))
  {
    fprintf(stderr, "SBAGLIATO!\n");
    printf("nostro: %d invece che %d\n", h_output, 1);
    exit(1);
  }
  else
  {
  	printf("risultato: %d == %d\n", h_output, 1);
  }
  
  cudaFree(d_input);
  cudaFree(d_output);
  return 0;
}

