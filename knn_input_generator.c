#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<math.h>
#include<string.h>

#define POINTS_MIN -100
#define POINTS_MAX 200

void generateRandomCoords(float* x, int spacedim, int numels);
void init(float* coords, float* coordsnew, int* classes, int spacedim, int numels, int classes_num, int newels);
void writeInput(float* coords, float* coordsnew, int* classes, int spacedim, int classes_num, int numels, int newels);

int main(int argc, char** argv)
{
  
  if (argc < 5)
  {
    perror("use: knn-generator <number-of-samples> <number-of-new-points> <number-of-classes> <space-dimension>");
    exit(1);
  }
  
  srand(time(NULL));
  int numels = atoi(argv[1]);          			   //number of existing points
  int newels = atoi(argv[2]);                      //number of points we want classify
  int classes_num = atoi(argv[3]);                 //number of classes
  int spacedim = atoi(argv[4]);                    //dimension of space
  float* coords;   		                       //coords of existing points with a class
  float* coordsnew;               			   		//coords of points we want classify
  int* classes;                                    //array with a class foreach points
  int totalElements = numels+newels;
  
  //*** allocation ***
  coords = malloc(sizeof(float)*totalElements*spacedim);
  coordsnew = malloc(sizeof(float)*newels*spacedim);
  classes = malloc(sizeof(int)*(totalElements));
  //*** end-allocation ***
    
  init(coords, coordsnew, classes, spacedim, numels, classes_num, newels);
  writeInput(coords, coordsnew, classes, spacedim, classes_num, numels, newels);
  return 0;
}

void init(float* coords, float* coordsnew, int* classes, int spacedim, int numels, int classes_num, int newels)
{
  int i;
  for (i = 0; i < numels; i++)
    classes[i] = rand()%classes_num;
  
  generateRandomCoords(coords, spacedim, numels);
  generateRandomCoords(coordsnew, spacedim, newels);
}

void generateRandomCoords(float* x, int spacedim, int numels)
{
  int i;
  for (i = 0; i < numels; i++)
  {
    int j;
    for (j = 0; j < spacedim; j++)
    	x[i*spacedim+j] = (float)rand()/(float)(RAND_MAX/POINTS_MAX) + (float)(POINTS_MIN);
  }
}

void writeInput(float* coords, float* coordsnew, int* classes, int spacedim, int classes_num, int numels, int newels)
{
  FILE *fp;
  fp = fopen("data", "w");
  int i, j;
  fprintf(fp, "%d,%d,%d,%d\n", numels, newels, classes_num, spacedim);
  for(i=0; i<numels; i++)
  {
  	for (j = 0; j < spacedim; j++)
  		fprintf(fp, "%lf,", coords[i*spacedim+j]);
  	fprintf(fp, "%d\n", classes[i]);
  }
   
  for(i = 0; i < newels; i++)
  {
  	for (j = 0; j < spacedim; j++)
  		fprintf(fp, "%lf,", coordsnew[i*spacedim+j]);
  	fprintf(fp, "-1\n");
  }
  fclose(fp);   
}
