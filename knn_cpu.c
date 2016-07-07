
/*
  C Implementation of KNN-Search algorithm
  we use mutli-dimensional points and euclidean distance
*/
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<string.h>
#include<time.h>

#define EXP 2
#define OUTPUT_FILE "ocpu"
#define INPUT_FILE "data"

float distance(float* coords, float* coords2, int spacedim);
int findClass(float* newcords, float* coords, int* classes, int spacedim, int classes_num, int newels, int numels, int k);
void swap_float(float* v, int x, int y);
void swap_int(int* v, int x, int y);
void kselectionsort(float* v, int* classes, int n, int k);
void writeOutput(float* coords, int* classes, int spacedim, int numels);
void readInput(FILE* file, float* coords, float* coordsnew, int* classes, int spacedim, int numels, int newels);
void simpleWriteOutput(int* classes, int numels);
void timespec_diff(const struct timespec *start, const struct timespec *end, struct timespec *ret);
float runtime_ms(const struct timespec *start,  const struct timespec *end);
void printStats(size_t numbytes, struct timespec before, struct timespec after, const char *msg);

int main(int argc, char** argv)
{  
  int newels;                      //number of points we want classify
  int k;                           //number of nearest points we use to classify
  int spacedim;
  int numels;                      //total element already classified
  char filePath[255];         //path + filname of input file
  int classes_num;                 //number of classes
  float* coords;          //coords of existing points with a class
  float* coordsnew;    //coords of points we want classify
  int* classes;                        //array with a class foreach points
  
  struct timespec before_time, after_time;
  float runtime;
  size_t numbytes;
  
  if (argc > 2) {
    strcpy(filePath, argv[1]);
    k = atoi(argv[2]);
  } else {
    printf("syntax err: knn <inputfile> <k> \n");
    exit(1);
  }  
   
  //read numels, newls, number-of-classes
  FILE *fp;
  if((fp = fopen(filePath, "r")) == NULL) {
        printf("No such file\n");
        exit(1);
  }
  
  fseek(fp, 0L, SEEK_END);
  float fileSize = ftell(fp);
  rewind(fp);
  
  fscanf(fp, "%d,%d,%d,%d\n", &numels, &newels, &classes_num, &spacedim);
  int totalElements = numels + newels;

  //*** allocation ***
  clock_gettime(CLOCK_MONOTONIC, &before_time);
  coords = malloc(sizeof(float)*totalElements*spacedim);
  coordsnew = malloc(sizeof(float)*newels*spacedim);     
  classes = malloc(sizeof(int)*totalElements);
  clock_gettime(CLOCK_MONOTONIC, &after_time);
  
  int i, j;  
  numbytes = (totalElements+newels)*(1+spacedim)*sizeof(float) + totalElements*sizeof(int);
  printStats(numbytes, before_time, after_time, "[time] allocation");
  //*** end-allocation *** 
  
  clock_gettime(CLOCK_MONOTONIC, &before_time);
  readInput(fp, coords, coordsnew, classes, spacedim, numels, newels);
  clock_gettime(CLOCK_MONOTONIC, &after_time);
  printStats(fileSize, before_time, after_time, "[time] read input file");
  
  fclose(fp);
  
  clock_gettime(CLOCK_MONOTONIC, &before_time);
  numbytes=0;
  for (i = 0; i < newels; i++)
  {
    classes[numels] = findClass(coordsnew+i*spacedim, coords, classes, spacedim, classes_num, newels, numels, k);
    
    for (j = 0; j < spacedim; j++)
      coords[numels*spacedim+j] = coordsnew[i*spacedim+j];
      
    numels++;
  }
  clock_gettime(CLOCK_MONOTONIC, &after_time);
  numbytes = (spacedim*totalElements*sizeof(float) + totalElements*sizeof(int))*newels;
  printStats(numbytes, before_time, after_time, "[time] knn algorithm");

  clock_gettime(CLOCK_MONOTONIC, &before_time);
  writeOutput(coords, classes, spacedim, numels);
  clock_gettime(CLOCK_MONOTONIC, &after_time);
  
  if((fp = fopen(filePath, "r")) == NULL) {
        printf("No such file\n");
        exit(1);
  }
  
  fseek(fp, 0L, SEEK_END);
  fileSize = ftell(fp);
  close(fp);
  
  printStats(fileSize, before_time, after_time, "[time] write output file");
  
  return 0;
}

float distance(float* coords, float* coords2, int spacedim)
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
 

int findClass(float* newcords, float* coords, int* classes, int spacedim, int classes_num, int newels, int numels, int k)
{      
  int i;
  float* distances = malloc(sizeof(float)*numels);
  int* classesCpy = malloc(sizeof(int)*(numels+newels));
  memcpy(classesCpy, classes, sizeof(int)*(numels+newels));
  
  //Part 1: init distances array
  for (i = 0; i < numels; i++){
    distances[i] = distance(coords+i*spacedim, newcords, spacedim);
  }
    
  //Part 2: find the k minimum distances and sort out distances and classesCpy arrays
  kselectionsort(distances, classesCpy, numels, k);
  
  //Part 3: select the class with the largest number of elements 
  int* classCount = malloc(sizeof(int)*classes_num);
  for (i = 0; i < classes_num; i++)
    classCount[i] = 0;
 
  for (i = 0; i < k; i++)
    classCount[classesCpy[i]]++;
     
  int max = 0;
  int maxValue = classCount[0];
  for (i = 1; i < classes_num; i++)
  {
    int value = classCount[i];
    if (value > maxValue)
    {
       max = i;
      maxValue = value;
    }
    else if (value != 0 && maxValue == value)
    {
        /*
          Classes have the same number of element (ambiguity)
          chooses the closest element class
        */
        int j = 0;
        for (j = 0; j < k; j++)
        {
          if (classesCpy[j] == i)
          {
              max = i;
              break;
          }
          else if (classesCpy[j] == max)
            break;
        }   
    }
  }
  
  free(classesCpy);
  free(distances);
  return max;
 }
 
 
void inline swap_float(float* v, int x, int y) {
  float t = v[x];
  v[x] = v[y];
  v[y] = t;
}

void inline swap_int(int* v, int x, int y) {
  int t = v[x];
  v[x] = v[y];
  v[y] = t;
}

//Find k-minimum in distance array
void kselectionsort(float* v, int* classes, int n, int k) {
  int i;
  for(i=0; i<k; i++) {
        int min = i;
        int j;
        for(j=i+1; j<n; j++) {
            if(v[j] < v[min])
                min = j;
        }
        swap_float(v, i, min);
        swap_int(classes, i, min);
   }
}

// read input from file
void readInput(FILE* file, float* coords, float* coordsnew, int* classes, int spacedim, int numels, int newels)
{
  int i, j;
  for(i=0; i<numels; i++)
  {
    for (j = 0; j < spacedim; j++)
      fscanf(file, "%f,", &(coords[i*spacedim+j]));
    fscanf(file, "%d\n", &(classes[i]));
  }
   
  for(i = 0; i < newels; i++)
  {
    for (j = 0; j < spacedim; j++)
      fscanf(file, "%f,", &(coordsnew[i*spacedim+j]));
    fscanf(file, "-1\n");
  }
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
    {
      fprintf(fp, "%f,", coords[i*spacedim+j]);
    }
    fprintf(fp, "%d\n", classes[i]);
  }
  fclose(fp); 
}

//Write classes array on file. Use it to compare results with cuda
void simpleWriteOutput(int* classes, int numels)
{
  FILE *fp;
  fp = fopen(OUTPUT_FILE, "w");
  int i;
  for(i = 0; i < numels; i++)
    fprintf(fp, "%d\n", classes[i]);    
  fclose(fp); 
}

//Time between timespec
void timespec_diff(const struct timespec *start,
  const struct timespec *end,
  struct timespec *ret)
{
  ret->tv_sec = end->tv_sec-start->tv_sec;
  if (end->tv_nsec<start->tv_nsec) {
    ret->tv_sec -= 1;
    ret->tv_nsec = 1000000000+end->tv_nsec-start->tv_nsec;
  } else {
    ret->tv_nsec = end->tv_nsec-start->tv_nsec;
  }
}

//Time between timespecs, in milliseconds
float runtime_ms(const struct timespec *start,
  const struct timespec *end) {
  struct timespec diff;
  timespec_diff(start, end, &diff);
  return diff.tv_sec*1000.0 + diff.tv_nsec/1.0e6;
}

void printStats(size_t numbytes, struct timespec before, struct timespec after, const char *msg)
{ 
  float runtime = runtime_ms(&before, &after);
  printf("%s %gms, %g GB/s\n", msg, runtime, numbytes/runtime/(1024*1024));
}
