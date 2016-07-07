K-Nearest Neighbor
===================
 A CUDA implementation
-------------------
##### Orazio Contarino and Emanuele Viglianisi
___

### Abstract
Pattern recognition is relevant in Data Analysis. Working with Big Data means that both computational time and hardware requirements are a real problem. We studied an implementation of K-Nearest Neighbor algorithm in CUDA in order to use the parallelization concept to solve classification problem. We propose different CUDA implementation and we also developed a C program for testing them. These programs use a CSV file as input, which contains points' classes and its spatial coordinates.

### Introduction
In Pattern Recognition, K-Nearest Neighbor is an algorithm used for the classification of points in a multidimensional space.  
Let's consider *numels* points with an assigned class and *newels* points which have to be classified. We want to assign a class each newels-point based on the k nearest neighbor. To find neighbors we use [Euclidean distance][1] and after that we choose the class with the highest frequency ([statistic mode][2]).

### Test first
#### CSV input generator
To test our implementation, we wrote a generator of CSV data. It's a command-line utility that allows to generate random coordinates in [-100.0, 100.0] and classes in [0, *number-of-classes*] as specified with the input args.  
How to use it:
```
./generator <number-of-training-samples> <number-of-new-points> <number-of-classes> <space-dimension>
```
#### CSV input file
The generator described above generates a file **data** that contains knn input in CSV format like this:
```
5,2,5,3
28.604126,59.192734,51.113876,0
-73.123611,7.033615,62.342010,4
-17.843704,64.565491,-34.666321,4
-9.518784,25.151924,92.620850,2
59.791962,-40.688431,97.609085,2
-47.902969,18.415474,-41.363827,-1
94.361481,84.595779,-59.809090,-1
```
The first line is a header that contains the number of numels, newels, classes, and space dimension.
Other lines are the body of the CSV data file where each row represents a point with its coordinates and class.
Class value "-1" means that the point has no class and so we have to assign it.

#### C implementation
C program read *data* input file. For each point, the **findClass()** function assigns a class.
Once every point has been classified, **writeOutput()** writes the result file **ocpu**, a csv file similar to the one we described above, in which every class has been determined.
We are not going to discuss the C implementation, as not particularly interesting, and only used a reference to check the results of the CUDA implementation.

#### Compare CUDA output and C output
Both C and CUDA programs write the result in file, *ocpu* and *ocuda*. To compare them we used Unix/Linux command line tools **sort** and **diff**.
```
sort ocuda > tmpcuda && sort ocpu > tmpcpu ; diff tmpcuda tmpcpu
```
If the two file match, no lines are shown in console.
******
### CUDA implementations
#### Reduction
First of all, we tried to parallelize the C algorithm using the CUDA library.
We used reduction to search for the minimum distance between the point and any others with an assigned class. We call reduction the general process of taking an input array (the points) and performing some computations that produce a smaller array of results (the shortest distance). This is done by calling the **findClass()** and **findMin()** kernels.

```c
...
for (i = 0; i < newels; i++)
{
  numBlocks = (numels + blockSize - 1)/blockSize;
  for (j = 0; j < k; j++)
  {
    findClass<<<numBlocks, blockSize, blockSize*4*sizeof(double)>>>(
    d_coords, d_coordsnew, d_classes,
    d_output,
    spacedim, classes_num,
    numels, j, i);

    findMin<<<1, blockSize, blockSize*4*sizeof(double)>>>(d_output, d_coords, d_coordsnew, d_classes, classes_num, spacedim, numels, j, d_result, k, i, numBlocks);
  }
  numels++;
}
...
```
The listing above is the Host code, a loop that for each point launches k reductions in order to find k nearest neighbors.
We used an offset to let the reduction ignore the **j** minimum already found. When every k minimum has been found, **findMin** will call the device function **findMode** that returns the class with the highest frequency.

We used shared memory to store the partial minimum array for each block in order to implement two-step reduction.
**findClass** is the first step which stores in global memory the minimum distance for each block. After that we are able to perform the last step by launching findMin kernel, getting the minimum value.

##### Conclusion
The choice of "the search of the minimum distance" as the main goal of the parallelization was not so good in term of performances. We exploit CUDA and its features only in the nearest neighbor search phase. Anyway, it isn't good because for each point the host has to launch *k* sequentially reductions. It's possible to parallelize in a more efficient way the knn algorithm, as proven by knn_punisher versions.

#### KNN_PUNISHER_V0
It's the first version of a different idea of parallelization for the knn algorithm. This time we want to parallelize the calculation of the k distances and not only the search of the minimum distance. We launch a kernel **knn** in which each thread is assigned to a single newels point and has the duty to build an array with its k nearest neighbor. Once the arrays have been determined the threads also try to assign a class. The threads can assign a class if and only if the array of neighbors contains only points with an assigned class. The output arrays are stored in global memory (**d_kOutput**) and then they are used by the **knnPunisher** kernel. At this point, there may be points we were not able to classify due to their dependencies. So now we have to "punish" them! The **knnPunisher** kernel is called to resolve all the points it can. In the worst case each point has a dependency with its previous. In this case, to assign a class to each newels the **knnPunisher** has to be called newels times.
```c
...
  knn<<<numBlocks, blockSize>>>(d_coords, d_kOutput, ...);
  for (int i = 0; i < newels; i++)
  {
      knnPunisher<<<numBlocks, blockSize, newels*sizeof(int)*2>>>(d_kOutput, d_classes, ...);
  }
...
```

Let's take a deep look at how we build the k nearest neighbor arrays. We explain the implementation in the following snippet code:

```c
__global__ void knn(float* coords, float2* kOutput, ...)
{
  ...
  for (i = 0; i < numels; i++)
  {
    pointCoords = coords+spacedim*i;

     //we calc distance between newpoint(threads associated newels) and every numels point
    float2 dist = make_float2(distance(newPointCoords, pointCoords, spacedim), i);

    //we try to insert, in an orderly manner, "dist" in kPoints (array of k-distances)
    insert(kPoints, dist, &size, k, gid);
  }

  /*
    We do the same for the newels point
    but here we count if we have added a newels
  */
  for (count=0; i < gid; i++)
  {
      pointCoords = coords+spacedim*i;
      float2 dist = make_float2(distance(newPointCoords, pointCoords, spacedim), i);
      count += insert(kPoints, dist, &size, k, gid);
  }

  if (count > 0)
  {
    //I couldn't assign a class due to a dependency with a previous "newels"
    classes[gid] = -1;
  }
  else
  {
    //All points in kArrays have an assigned class - we can assign a class
    int kclasses[KMAX];
    for (int j = 0; j < k; j++){
      kclasses[j] = classes[(int)(kPoints[j].y)];
    }

    int newClass = deviceFindMode(kclasses, classes_num, k);
    classes[gid] = newClass;
  }

  //copy kPoints in kOutput
  int newelId = gid-numels;
  for (i = 0; i < k; i++)
    kOutput[newelId*KMAX + i] = kPoints[i];
}

//KNN PUNISHER KERNEL
__global__ void knnPunisher(float2* kOutput, int* classes, ...)
{
  ...
  /*
    To reduce global memory access, we decided to copy
    all previous "classes value" in shared memory.
  */
  while (i < gid)
  {
    mPartial[i] = classes[i+numels];
    i += blockDim.x;
  }

  if (gid < newels)
    mPartial[gid] = classes[gid+numels];

  if (gid >= newels || mPartial[gid] != -1) return;

  //If i am here, we have to assign a class to the thread's newels
  int kPoints[KMAX];
  for (int i = 0; i < k; i++)
    kPoints[i] = kOutput[gid*KMAX+i].y;

  //Here we check if dependencies have already an assigned class
  int count = 0;
  for (int i = k-1; i >= 0; i--)
  {
    int id = kPoints[i];
    int lid = id - numels;
    if (id > numels && mPartial[lid] < 0)
    {
      //there is a point with no class
      //so we can't set a class yet
      count++;
      break;
    }
  }

  if (count == 0)
  {
    //All dependencies have a class
    //A class can be assigned to the thread's newels
    int kclasses[KMAX];
      for (int j = 0; j < k; j++)
          kclasses[j] = classes[kPoints[j]];
    int newClass = deviceFindMode(kclasses, classes_num, k);
    classes[gid+numels] = newClass;
  }
}
```
##### Conclusion
This time much more work has been parallelized, limiting the presence of sequential code to only the calculation of the distance and its insertion into kPoints array. Anyway, we force the kernel to be launched a fixed number (*newels*). So we may continue to launch the kernel although every points' class has been already assigned. In the following versions we fixed this issue comparing the performance.

#### KNN_PUNISHER_V1
To avoid an excessive number of kernel calls, we use findIntInArray kernel which allows to verify the presence of newels not currently classified. We copy the result from device to host, to let the host detect when every element has a class. At this point no more kernel will be launched.

```c
knn<<<numBlocks, blockSize>>>(d_coords, d_kOutput, ...);
int again = 1;
while (again != 0)
{
  knnPunisher<<<numBlocks, blockSize, newels*sizeof(int)*2>>>(d_kOutput...);
  check_error(cudaMemset(d_again, 0, sizeof(int)), "reset d_again");
  findIntInArray<<<numBlocks, blockSize>>>(d_classes+numels, newels, d_again);
  check_error(cudaMemcpy(&again, d_again, sizeof(int), cudaMemcpyDeviceToHost), "download d_again");
}
 ```
##### Conclusion
We fixed the problem described in the previous version with findIntInArray kernel. However, we introduced inside a loop a new kernel launch and a device-to-host copy that affect performances.

#### KNN_PUNISHER_V1_SELF
We implemented findIntInArray inside knnPunisher avoiding an unnecessary kernel launch.

```c
int again = 1;
while (again != 0)
{
  check_error(cudaMemset(d_again, 0, sizeof(int)), "reset d_again");
  knnPunisher<<<numBlocks, blockSize, newels*sizeof(int)*2>>>(d_kOutput, ..., d_again);
  check_error(cudaMemcpy(&again, d_again, sizeof(int), cudaMemcpyDeviceToHost), "download d_again");
}
```

##### Conclusion
Though we avoid an unnecessary kernel launch, the problem of the device-to-host copy inside a loop persists.

#### KNN_PUNISHER_ROLL
The kernel launch loop has been replaced by threads loop in knnPunisher kernel avoiding the sequential kernel launch and the device-to-host copy.

```c
//HOST
knn<<<numBlocks, blockSize>>>(d_coords, d_kOutput,...);
knnPunisher<<<numBlocks, blockSize, newels*sizeof(int)*2>>>(d_kOutput,...);

//KNNPUNISHER
__global__ void knnPunisher(float2* kOutput, ...)
{
  ...
  //Threads loop until the point's class is assigned
  while(count != 0)
  {
      //update shared-memory with latest assigned classes
      for (i = 0; i < gid; i++)
            mPartial[i] = classes[i+numels];

      mPartial[gid] = classes[gid+numels];
      if (mPartial[gid] != -1) return;
      __syncthreads();

      //Here we check if dependencies have already an assigned class
      count = 0;
      for (i = k-1; i >= 0; i--)
      {
        id = kPoints[i];
        lid = id - numels;
        if (id > numels && mPartial[lid] < 0)
        {
          count++;
          break;
        }
      }

      if (count == 0)
      {
        ...
      }
  }
}
```
##### Conclusion
Removing the loop from the host, improve (just a little) the performances.

#### KNN_PUNISHER_ROLL_OPTIMIZED
In this version we tried to force the compiler to use registers and local memory by giving at compile time the dimension of arrays, changing the memory access type from global to local (where possible). We also set kernel parameters with the directives *const* and *restrict* in order to permit the compiler to optimize as much as it is able to.

KNN function does a lot of accesses in global memory and this is the bottleneck of the knn-punisher versions.
To improve performance of knn kernel we stored each *kPoints* array in a dedicated bank (coloumn) of shared memory avoiding bank conflicts.

```c
#define SPACEDIMMAX 300
...

...
__global__ void knn(float* const restrict coords, float2* restrict kOutput, const int totalElements, const int numels, const int spacedim, const int k, int* restrict classes, const int classes_num);
__global__ void knnPunisher(float2* restrict kOutput, int* restrict classes, const int numels, const int newels, const int k, const int classes_num);
...

...
//KNN[1]
 for (int i = 0; i < spacedim; i++)
    point[i] = newPointCoords[i];
...

...
//KNN[2]
  pointCoords = coords;
  for (i = 0; i < numels; i++)
  {
    dist = make_float2(distance(point, pointCoords, spacedim), i);
    insert(shm+lid, dist, &size, k, gid, offset);
    pointCoords += spacedim;
  }
...

```
##### Conclusion
There are significant changes in performance compared to the previous KNN_PUNISHER_ROLL. KNN_PUNISHER_ROLL doesn't help the compiler with the directives and doesn't use the shared memory in knn kernel at all.

### Final conclusion and comparison
There are no significant changes in performance among all knn_punisher versions. The only one which got a great boost was knn_punisher_optimized caused by the download of threads' point's coords from global memory to private memory. KNN_PUNISHER_ROLL_OPTIMIZED is the best choice, by the way this version has several limits bacause we define at compile time the max size of classes and spacedim. If a greater size is needed, define values have to be changed, considering architecture's limits.
It is also important to observe that knn kernel takes a very long time. This may be a problem with a huge input because watchdog daemon kills every (very) long task.

### References
1. http://mathworld.wolfram.com/Distance.html
2. https://statistics.laerd.com/statistical-guides/measures-central-tendency-mean-mode-median.php

[1]: http://mathworld.wolfram.com/Distance.html
[2]: https://statistics.laerd.com/statistical-guides/measures-central-tendency-mean-mode-median.php
