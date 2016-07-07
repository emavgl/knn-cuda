#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<float.h> //DBL_MAX
#include <cuda_runtime_api.h>

#define restrict __restrict__
#define PADDINGCLASS -2
#define OUTPUT_FILE "ocuda"
#define INPUT_FILE "data"
#define KMAX 20
#define SPACEDIMMAX 150
#define CLASSESMAX 100

void check_error(cudaError_t err, const char *msg);
void printStats(cudaEvent_t before, cudaEvent_t after, const char *msg);
void readInput(FILE* file, float* coords, int* classes, int spacedim, int numels, int totalElements);
void writeOutput(float* coords, int* classes, int spacedim, int numels);

__global__ void knn(float* coords, float2* kOutput, int totalElements, int numels, int spacedim, int k, int* classes, int classes_num);
__global__ void knnPunisher(float2* kOutput, int* d_classes, int numels, int newels, int k, int classes_num);

__device__ float distance(float* coords, float* coords2, int spacedim);
__device__ int insert(float2* kPoints, float2 newDist, int* size, int k, int gid);
__device__ void swapfloat2(float2* d1, float2* d2);
__device__ int deviceFindMode(int* kclasses, int classes_num, int k);
__device__ float distanceShm(float* coords, int left, int spacedim);

//Declaration of shared-memory. It's going to contains partial minimum of distances
extern __shared__ int mPartial[];

int main(int argc, char *argv[])
{  
  int newels;                      //number of points we want classify
  int k;                           //number of nearest points we use to classify
  int numels;                      //total element already classified
  int spacedim;
  char filePath[255];              //path + filname of input file
  int classes_num;                 //number of classes
  float* h_coords;                //coords of existing points with a class
  int* h_classes;                  //array contains the class for each points
  
  //*** Device-variables-declaration ***
  float* d_coords;
  int2* d_determinate;
  int* d_classes;
  float2* d_kOutput;
  //*** end-device-declaration
  
  //***cudaEvent-declaration***
  cudaEvent_t before_allocation, before_input, before_upload, before_knn, before_download;
  cudaEvent_t after_allocation, after_input, after_upload, after_knn, after_download;
  //***end-cudaEvent-declaration***
  
  //Requisiti: numels e newels devono essere maggiori di K
  if (argc > 2) 
  {
    strcpy(filePath, argv[1]);
    k = atoi(argv[2]);
  }
  else 
  {
    printf("how-to-use: knn <inputfile> <k> \n");
    exit(1);
  } 
  
  //***cuda-init-event***
  check_error(cudaEventCreate(&before_allocation), "create before_allocation cudaEvent");
  check_error(cudaEventCreate(&before_input), "create before_input cudaEvent");
  check_error(cudaEventCreate(&before_upload), "create before_upload cudaEvent");
  check_error(cudaEventCreate(&before_knn), "create before_knn cudaEvent");
  check_error(cudaEventCreate(&before_download), "create before_download cudaEvent");
  
  check_error(cudaEventCreate(&after_allocation), "create after_allocation cudaEvent");
  check_error(cudaEventCreate(&after_input), "create after_input cudaEvent");
  check_error(cudaEventCreate(&after_upload), "create after_upload cudaEvent");
  check_error(cudaEventCreate(&after_knn), "create after_knn cudaEvent");
  check_error(cudaEventCreate(&after_download), "create after_download cudaEvent");
  //***end-cuda-init-event***

  FILE *fp;
  if((fp = fopen(filePath, "r")) == NULL)
  {
        printf("No such file\n");
        exit(1);
  }
  
  fseek(fp, 0L, SEEK_END);
  float fileSize = ftell(fp);
  rewind(fp);
  
  int count = fscanf(fp, "%d,%d,%d,%d\n", &numels, &newels, &classes_num, &spacedim);
  int totalElements = numels + newels;

  //*** allocation ***
  cudaEventRecord(before_allocation);
  h_coords = (float*) malloc(sizeof(float)*totalElements*spacedim);
  h_classes = (int*) malloc(sizeof(int)*totalElements);
   
  //*** device-allocation ***
  check_error(cudaMalloc(&d_coords, totalElements*spacedim*sizeof(float)), "alloc d_coords_x");
  check_error(cudaMalloc(&d_classes, totalElements*sizeof(int)), "alloc d_classes");
  check_error(cudaMalloc(&d_determinate, newels*2*sizeof(int)), "alloc d_determinate");
  check_error(cudaMalloc(&d_kOutput, newels*KMAX*2*sizeof(float)), "alloc d_kOutput");

  //*** end-device-allocation ***
  cudaEventRecord(after_allocation);
  
  ///***input-from-file***
  cudaEventRecord(before_input);
  readInput(fp, h_coords, h_classes, spacedim, numels, totalElements);
  cudaEventRecord(after_input);
  fclose(fp);
  ///***end-input-from-file***

  //***copy-arrays-on-device***
  cudaEventRecord(before_upload);
  check_error(cudaMemcpy(d_coords, h_coords, totalElements*spacedim*sizeof(float), cudaMemcpyHostToDevice), "copy d_coords");
  check_error(cudaMemcpy(d_classes, h_classes, totalElements*sizeof(int), cudaMemcpyHostToDevice), "copy d_classes");
  cudaEventRecord(after_upload);
  //***end-copy-arrays-on-device***                              
  
  const int blockSize = 512;
  int numBlocks = (newels + blockSize - 1)/blockSize;
   
  cudaEventRecord(before_knn);
  knn<<<numBlocks, blockSize>>>(d_coords, d_kOutput, totalElements, numels, spacedim, k, d_classes, classes_num);
  for (int i = 0; i < newels; i++)
  {
      knnPunisher<<<numBlocks, blockSize, newels*sizeof(int)*2>>>(d_kOutput, d_classes, numels, newels, k, classes_num);
  }
  cudaEventRecord(after_knn);
  check_error(cudaMemcpy(h_classes+numels, d_classes+numels, newels*sizeof(int), cudaMemcpyDeviceToHost), "download classes");

  check_error(cudaEventSynchronize(after_knn), "sync cudaEvents");
  printStats(before_knn, after_knn, "knn");
    
  writeOutput(h_coords, h_classes, spacedim, totalElements);
  return 0;
}

void check_error(cudaError_t err, const char *msg)
{
  if (err != cudaSuccess) 
  {
    fprintf(stderr, "%s : error %d (%s)\n", msg, err, cudaGetErrorString(err));
    exit(err);
  }
}

float runtime;
void printStats(cudaEvent_t before, cudaEvent_t after, const char *msg)
{ 
  check_error(cudaEventElapsedTime(&runtime, before, after), msg);
  printf("%s %gms\n", msg, runtime);
}

__global__ void knn(float* coords, float2* kOutput, int totalElements, int numels, int spacedim, int k, int* classes, int classes_num)
{
  int gid = numels + threadIdx.x + blockIdx.x*blockDim.x; //id del punto da determinare
  if (gid >= totalElements) return;
  
  float* newPointCoords = coords+spacedim*gid;
  float* pointCoords;
  float2 kPoints[KMAX];

  int i = 0, size = 0, count = 0;
  float2 dist;
  
  for (i = 0; i < numels; i++)
  {
    pointCoords = coords+spacedim*i;
    dist = make_float2(distance(newPointCoords, pointCoords, spacedim), i);
    insert(kPoints, dist, &size, k, gid);
  }
  
  for (count=0; i < gid; i++)
  {
      pointCoords = coords+spacedim*i;
      dist = make_float2(distance(newPointCoords, pointCoords, spacedim), i);
      count += insert(kPoints, dist, &size, k, gid);
  }
    
  if (count > 0)
  {
    classes[gid] = -1;
  }
  else
  {
    int kclasses[KMAX];
    for (int j = 0; j < k; j++)
      kclasses[j] = classes[(int)(kPoints[j].y)];
    classes[gid] = deviceFindMode(kclasses, classes_num, k);
  }
  
  //copia kPoints in kOutput
  int newelId = gid-numels;
  for (i = 0; i < k; i++)
    kOutput[newelId*KMAX + i] = kPoints[i];
}

__global__ void knnPunisher(float2* kOutput, int* classes, int numels, int newels, int k, int classes_num)
{
  int gid = threadIdx.x + blockIdx.x*blockDim.x;
  int lid = threadIdx.x;
  
  int i = lid;  
  while (i < gid)
  {
    mPartial[i] = classes[i+numels];
    i += blockDim.x;
  }
  
  if (gid < newels)
    mPartial[gid] = classes[gid+numels];
  
  if (gid >= newels || mPartial[gid] != -1) return;
  
  //Se sono qui la classe per il kPoint è da determinare
  int kPoints[KMAX];
  for (int i = 0; i < k; i++)
    kPoints[i] = kOutput[gid*KMAX+i].y; //gid
  
  //Le sue dipendenze, se già determinate stanno nella shared-memory
  int count = 0;
  for (int i = k-1; i >= 0; i--)
  {
    int id = kPoints[i];
    int lid = id - numels;
    if (id > numels && mPartial[lid] < 0)
    {
      //segno quelli indeterminati
      count++;
      break;
    }
  }
  
  if (count == 0)
  {
    //posso determinare il punto
    //le sue dipendenze si trovano in shared memory
    int kclasses[KMAX];
      for (int j = 0; j < k; j++)
          kclasses[j] = classes[kPoints[j]];
    int newClass = deviceFindMode(kclasses, classes_num, k);
    classes[gid+numels] = newClass;
  }
}

__device__ int deviceFindMode(int* kclasses, int classes_num, int k)
{
  int classesCount[CLASSESMAX];

  int i;
  int temp=0;
  
  for (i = 0; i < CLASSESMAX; i++)
    classesCount[i] = 0;
       
  for (i = 0; i < k; i++){
    temp=kclasses[i];
    classesCount[temp]+=1;
  } 

  int max = 0;
  int maxValue = classesCount[0];

  for (i = 1; i < classes_num; i++)
  {
    int value = classesCount[i];
    if (value > maxValue)
    {
      max = i;
      maxValue = value;
    }
    else if (value != 0 && maxValue == value)
    {
        int j = 0;
        for (j = 0; j < k; j++)
        {
          if (kclasses[j] == i)
          {
            max = i;
            break;
          }
          else if (kclasses[j] == max)
            break;
        }
    }
  }
  
  return max;
}

//inserimento smart in kPoints
__device__ int insert(float2* kPoints, float2 newDist, int* size, int k, int gid)
{  
    int inserted = 0;
    if (*size == 0)
    {
      //Caso base: inserimento su array vuoto
      kPoints[0] = newDist;
      *size = *size + 1;
      return 1;
    }
  
    int i = 1;
    float2* value = &newDist; //nuovo elemento
    float2* tail = &(kPoints[*size-i]);
   
    if (*size < k)
    {  
        kPoints[*size] = newDist;
        value = &(kPoints[*size]);
        inserted = 1;
    }

    //partire della fine, swap se trovo elemento più grande - mi fermo se trovo elemento più piccolo
    while (i <= *size && (*tail).x > (*value).x)
    {
        swapfloat2(tail, value);
        value = tail;
        i++;
        tail = &(kPoints[*size-i]);
        inserted = 1;    
    }
    
    if (inserted && *size < k) *size = *size + 1;
    return inserted;
}

__device__ void swapfloat2(float2* d1, float2* d2)
{
  //da provate tmp = d1, d1 = d2, d2 = tmp
  float2 tmp;
  tmp.x = (*d1).x;
  tmp.y = (*d1).y;
  
  (*d1).x = (*d2).x;
  (*d1).y = (*d2).y;
  
  (*d2).x = tmp.x;
  (*d2).y = tmp.y;
}


// read input from file
void readInput(FILE* file, float* coords, int* classes, int spacedim, int numels, int totalElements)
{
  int i, j;
  int count;
  for(i=0; i<numels; i++)
  {
    for (j = 0; j < spacedim; j++)
      count = fscanf(file, "%f,", &(coords[i*spacedim +j]));
    count = fscanf(file, "%d\n", &(classes[i]));
  }
   
  for(; i < totalElements; i++)
  {
    for (j = 0; j < spacedim; j++)
      count = fscanf(file, "%f,", &(coords[i*spacedim+j]));
    count = fscanf(file, "-1\n");
  }
  count++;
}

//Write Output on file
void writeOutput(float* coords, int* classes, int spacedim, int numels)
{
  FILE *fp;
  fp = fopen(OUTPUT_FILE, "w");
  int i, j;
  for( i = 0; i < numels; i++)
  {
    for (j = 0; j < spacedim; j++)
      fprintf(fp, "%lf,", coords[i*spacedim+j]);
    
    fprintf(fp, "%d\n", classes[i]);
  }
  fclose(fp); 
}

//multidimensional euclidian distance (without sqrt)
__device__ float distance(float* coords, float* coords2, int spacedim)
{
  float sum = 0;
  int i;
  for (i = 0; i < spacedim; i++)
  {
    float diff = coords[i] - coords2[i];
    sum += diff*diff;
  }  
  return sum;
}
