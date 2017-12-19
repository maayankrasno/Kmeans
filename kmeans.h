#ifndef __KMEANS_H
#define __KMEANS_H

#pragma warning( disable : 4996 )
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include "kernel.h"
#include "structs.h"


MPI_Datatype makeMPIPoduct();
MPI_Datatype makeMPICluster(MPI_Datatype MPI_Product);
void broadCastAllParameters(int* numOfProducts, int* maxNumOfIterations, int* numOfDimensions, int* maxNumOfClusters);
void giveProductToProcs(Product* allProducts, int numOfProducts, Product** myProducts, int* myProductsSize, int numprocs, int myid, MPI_Datatype MPI_Product);
void recvProduct(Product* myProduct, MPI_Datatype MPI_Product, int source);
void sendProduct(Product* product, int address, MPI_Datatype MPI_Product);
void scatterProducts(Product* allProducts, int numOfProducts, int numprocs, MPI_Datatype MPI_Product);
void sendInformationToOtherProcesses(int* numOfProducts, int* numOfDimensions);
double evaluateQualityOfClusters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts);
double findDiameter(int clusterId, Product* allProducts, int numOfProducts);
Product* readDataFromFile(int* numOfPoints, int* numOfDimensions, int* maxNumOfClusters, int* maxNumOfIterations, double* wantedQualityOfClusters);
void readProductsFromFile(FILE* dataFile, Product* allProducts, int numOfProducts, int numOfDimensions);
void broadCastClusterToProcs(Cluster* allClusters, int numOfClusters, int myid, int numprocs, MPI_Datatype MPI_Cluster);
void kmeans(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts, Product* myProducts, int myProductSize, int maxNumOfIterations, int myid, int numprocs, MPI_Datatype MPI_Cluster, MPI_Datatype MPI_Product);
void checkChangeStatus(int* changed, int numprocs, int myid);
void gatherAllProducts(Product* allProducts, int numOfProducts, Product* myProducts, int myProductSize, int myid, MPI_Datatype MPI_Product);
void giveClustersVirtualCenters(Cluster* allClusters, int numOfClusters, Product* allProducts);
int setProductsToClusters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts);
double calculateDistance(Product* p1, Product* p2);
void setNewVirtualCenters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts);
void syncAndCalculateAvarage(Cluster* virtualCenters, Cluster* allClusters, int numOfClusters, int dimensions);

void handleErrors(cudaError_t cudaStatus, const char* errorMessage , void* ptr);
Cluster*  allocateClustersToGPU(Cluster* allClusters, int numOfClusters);
Product* allocateProductsToGPU(Product* allProducts, int numOfProducts);
void copyClustersFromGPU(Cluster* allClusters, Cluster* cudaClusters, int numOfClusters);
void copyProductsFromGPU(Product* allProducts, Product* cudaProducts, int numOfProducts);

#endif