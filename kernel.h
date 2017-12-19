#ifndef __KERNEL_H
#define __KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "structs.h"
#include "kmeans.h"

int setProductsToClustersKernel(Cluster* cudaClusters, int numOfClusters, Product* cudaProducts, int numOfProducts);
__global__ void cudaSetProductToCluster(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts, int* changed);
__device__ double cudaCalculateDistance(Product* p1, Product* p2);
#endif