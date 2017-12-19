#include "kmeans.h"
#include "kernel.h"

MPI_Datatype makeMPIPoduct()
{
	const int nitems = 4;
	int blocklengths[4] = { LEN, 1, 1, 1 };
	MPI_Datatype types[4] = { MPI_CHAR, MPI_INT, MPI_INT, MPI_DOUBLE };
	MPI_Datatype MPI_Product;
	MPI_Aint offsets[4];

	offsets[0] = offsetof(Product, name);
	offsets[1] = offsetof(Product, dimensions);
	offsets[2] = offsetof(Product, clusterId);
	offsets[3] = offsetof(Product, coordinates);

	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_Product);
	MPI_Type_commit(&MPI_Product);

	return MPI_Product;
}

MPI_Datatype makeMPICluster(MPI_Datatype MPI_Product)
{
	const int nitems = 2;
	int blocklengths[2] = { 1, 1 };
	MPI_Datatype types[2] = { MPI_INT, MPI_Product };
	MPI_Datatype MPI_Cluster;
	MPI_Aint offsets[2];

	offsets[0] = offsetof(Cluster, numOfProducts);
	offsets[1] = offsetof(Cluster, virutalCenter);


	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_Cluster);
	MPI_Type_commit(&MPI_Cluster);

	return MPI_Cluster;
}

void broadCastAllParameters(int* numOfProducts, int* maxNumOfIterations, int* numOfDimensions, int* maxNumOfClusters)
{
	MPI_Bcast(numOfProducts, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(maxNumOfIterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(numOfDimensions, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(maxNumOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void giveProductToProcs(Product* allProducts, int numOfProducts, Product** myProducts, int* myProductsSize, int numprocs, int myid, MPI_Datatype MPI_Product)
{
	int i, remainder;

	remainder = numOfProducts % numprocs;
	*myProductsSize = (numOfProducts / numprocs) + (myid == 0 ? remainder : 0);
	*myProducts = (Product*)calloc(*myProductsSize, sizeof(Product));

	if (myid == 0)
	{
#pragma omp parallel for
		for (i = 0; i < *myProductsSize; i++)
		{
			(*myProducts)[i] = allProducts[i];
		}
		scatterProducts(allProducts + (*myProductsSize), numOfProducts - *myProductsSize, numprocs, MPI_Product);
	}

	else
	{
		for (i = 0; i < *myProductsSize; i++)
		{
			recvProduct(&(*myProducts)[i], MPI_Product, 0);
		}
	}
}

void recvProduct(Product* myProduct, MPI_Datatype MPI_Product, int source)
{
	MPI_Status status;
	int dimensions;

	if (myProduct->coordinates != NULL)
	{
		free(myProduct->coordinates);
	}

	MPI_Recv(myProduct, 1, MPI_Product, source, 0, MPI_COMM_WORLD, &status);
	dimensions = myProduct->dimensions;
	myProduct->coordinates = (double*)calloc(sizeof(double), dimensions);
	MPI_Recv((*myProduct).coordinates, dimensions, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);

}

void sendProduct(Product* product, int address, MPI_Datatype MPI_Product)
{
	MPI_Send(product, 1, MPI_Product, address, 0, MPI_COMM_WORLD);
	MPI_Send(product->coordinates, product->dimensions, MPI_DOUBLE, address, 0, MPI_COMM_WORLD);
}


void scatterProducts(Product* allProducts, int numOfProducts, int numprocs, MPI_Datatype MPI_Product)
{
	int i;
	for (i = 0; i < numOfProducts; i++)
	{
		sendProduct(&allProducts[i], 1 + (i % (numprocs - 1)), MPI_Product);
	}
}

double evaluateQualityOfClusters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts)
{
	int i, j, counter = 0;
	double diameter, quality = 0;

#pragma omp parallel for reduction(+:quality), reduction(+:counter)
	for (i = 0; i < numOfClusters; i++)
	{
		diameter = findDiameter(i, allProducts, numOfProducts);

		for (j = 0; j < numOfClusters; j++)
		{
			if (i != j)
			{
				quality += diameter / calculateDistance(&allClusters[i].virutalCenter, &allClusters[j].virutalCenter);
				counter++;
			}
		}
	}
	return quality / counter;
}


double findDiameter(int clusterId, Product* allProducts, int numOfProducts)
{
	int i, j;
	double maxDiameter = 0, diameter;

	for (i = 0; i < numOfProducts; i++)
	{
		for (j = i + 1; j < numOfProducts && allProducts[i].clusterId == clusterId; j++)
		{
			if (allProducts[j].clusterId == clusterId)
			{
				diameter = calculateDistance(&allProducts[i], &allProducts[j]);
				if (diameter > maxDiameter)
				{
					maxDiameter = diameter;
				}
			}
		}
	}
	return maxDiameter;
}

Product* readDataFromFile(int* numOfPoints, int* numOfDimensions, int* maxNumOfClusters, int* maxNumOfIterations, double* wantedQualityOfClusters)
{
	FILE* dataFile = fopen("C:\\Users\\afeka\\Desktop\\parallel_computing\\parallel_computing\\Sales_Transactions_CorrectFormat.txt", "r");

	if (!dataFile)
	{
		printf("Opening the file has failed\n");
		MPI_Finalize();
		exit(1);
	}

	fscanf(dataFile, "%d%d%d%d%lf", numOfPoints, numOfDimensions, maxNumOfClusters, maxNumOfIterations, wantedQualityOfClusters);

	Product* allProducts = (Product*)malloc(sizeof(Product)**numOfPoints);
	readProductsFromFile(dataFile, allProducts, *numOfPoints, *numOfDimensions);

	fclose(dataFile);
	return allProducts;
}


void readProductsFromFile(FILE* dataFile, Product* allProducts, int numOfProducts, int numOfDimensions)
{
	int i, j;

	for (i = 0; i < numOfProducts; i++)
	{
		fscanf(dataFile, "%s", allProducts[i].name);
		allProducts[i].dimensions = numOfDimensions;
		allProducts[i].coordinates = (double*)malloc(sizeof(double)* numOfDimensions);
		allProducts[i].clusterId = NO_CLUSTER;

		for (j = 0; j < numOfDimensions; j++)
		{
			fscanf(dataFile, "%lf", &allProducts[i].coordinates[j]);
		}
	}
}
void broadCastClusterToProcs(Cluster* allClusters, int numOfClusters, int myid, int numprocs, MPI_Datatype MPI_Cluster)
{
	int i, j;
	MPI_Status status;
	MPI_Bcast(allClusters, numOfClusters, MPI_Cluster, 0, MPI_COMM_WORLD);

	if (myid == 0)
	{
		for (i = 0; i < numOfClusters; i++)
		{
			for (j = 1; j < numprocs; j++)
			{
				MPI_Send(allClusters[i].virutalCenter.coordinates, allClusters[i].virutalCenter.dimensions, MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
			}
		}
	}

	else
	{
		for (i = 0; i < numOfClusters; i++)
		{
			allClusters[i].virutalCenter.coordinates = (double*)calloc(sizeof(double), allClusters[i].virutalCenter.dimensions);
			MPI_Recv(allClusters[i].virutalCenter.coordinates, allClusters[i].virutalCenter.dimensions, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
		}
	}

}
void kmeans(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts, Product* myProducts, int myProductSize, int maxNumOfIterations, int myid, int numprocs, MPI_Datatype MPI_Cluster, MPI_Datatype MPI_Product)
{
	//changed and numOfIteration are exit parameters 
	int changed = 0, numOfIterations = 0;
	int isChanged = 0;
	//give each cluster a starting random virtual center
	if (myid == 0)
	{
		giveClustersVirtualCenters(allClusters, numOfClusters, allProducts);
	}
	//inform all processes what are the virtual centers
	broadCastClusterToProcs(allClusters, numOfClusters, myid, numprocs, MPI_Cluster);
	
	do
	{
		//1. for each point assign the cluster that it is the closest to (distance equation)
		changed = setProductsToClusters(allClusters, numOfClusters, myProducts, myProductSize);

		//2. check termination condition on changed variable
		MPI_Allreduce(&changed, &isChanged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		//3. gather the products from all procs
		gatherAllProducts(allProducts, numOfProducts, myProducts, myProductSize, myid, MPI_Product);

		//4. set new virtual center to the cluster (by doing avarage of the products belonging to it)
		setNewVirtualCenters(allClusters, numOfClusters, myProducts, myProductSize);

		numOfIterations++;

	} while (isChanged && numOfIterations <= maxNumOfIterations);
}

void gatherAllProducts(Product* allProducts, int numOfProducts, Product* myProducts, int myProductsSize, int myid, MPI_Datatype MPI_Product)
{
	int i;

	if (myid == 0)
	{
#pragma omp parallel for
		for (i = 0; i < myProductsSize; i++)
		{
			allProducts[i] = myProducts[i];
		}
		for (i = myProductsSize; i < numOfProducts; i++)
		{
			recvProduct(&allProducts[i], MPI_Product, MPI_ANY_SOURCE);
		}
	}

	else
	{
		for (i = 0; i < myProductsSize; i++)
		{
			sendProduct(&myProducts[i], 0, MPI_Product);
		}
	}
}

void giveClustersVirtualCenters(Cluster* allClusters, int numOfClusters, Product* allProducts)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < numOfClusters; i++)
	{
		if (allClusters[i].virutalCenter.coordinates != NULL)
			free(allClusters[i].virutalCenter.coordinates);
		allClusters[i].virutalCenter = allProducts[i];
		allClusters[i].virutalCenter.coordinates = (double*)malloc(sizeof(double)*allProducts->dimensions);
		memcpy(allClusters[i].virutalCenter.coordinates, allProducts[i].coordinates, sizeof(double)*allProducts[i].dimensions);
		allClusters[i].virutalCenter.clusterId = i;
	}
}


int setProductsToClusters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts)
{

	Cluster* cudaClusters;
	Product* cudaProducts;

	int changed = 0;

	cudaProducts = allocateProductsToGPU(allProducts, numOfProducts);
	cudaClusters = allocateClustersToGPU(allClusters, numOfClusters);

	changed = setProductsToClustersKernel(cudaClusters, numOfClusters, cudaProducts, numOfProducts);

	copyProductsFromGPU(allProducts, cudaProducts, numOfProducts);
	copyClustersFromGPU(allClusters, cudaClusters, numOfClusters);

	return changed;
}

double calculateDistance(Product* p1, Product* p2)
{
	int i, dimensions = p2->dimensions;
	double distance = 0;

	for (i = 0; i < dimensions; i++)
	{
		distance += (p1->coordinates[i] - p2->coordinates[i])*(p1->coordinates[i] - p2->coordinates[i]);
		//distance += pow(p1->coordinates[i] - p2->coordinates[i], 2);
	}

	distance = sqrt(distance);
	return distance;
}


void setNewVirtualCenters(Cluster* allClusters, int numOfClusters, Product* allProducts, int numOfProducts)
{
	int i, j, dimensions = allProducts[0].dimensions;

	// aids to calculate the new virtual center
	Cluster* virtualCenters = (Cluster*)calloc(numOfClusters, sizeof(Cluster));
	//Product* virtualCenters = (Product*)malloc(sizeof(Product)*numOfClusters);

	// allocates space for the coordinates
#pragma omp parallel for
	for (i = 0; i < numOfClusters; i++)
	{
		virtualCenters[i].virutalCenter.dimensions = dimensions;
		virtualCenters[i].virutalCenter.coordinates = (double*)calloc(dimensions, sizeof(double));
	}

	for (i = 0; i < numOfProducts; i++)
	{
		virtualCenters[allProducts[i].clusterId].numOfProducts++;
		for (j = 0; j < dimensions; j++)
		{
			virtualCenters[allProducts[i].clusterId].virutalCenter.coordinates[j] += allProducts[i].coordinates[j];
		}
	}
	syncAndCalculateAvarage(virtualCenters, allClusters, numOfClusters, dimensions);

#pragma omp parallel for
	for (i = 0; i < numOfClusters; i++)
		free(virtualCenters[i].virutalCenter.coordinates);
	free(virtualCenters);
}

void syncAndCalculateAvarage(Cluster* virtualCenters, Cluster* allClusters, int numOfClusters, int dimensions)
{
	int i, j;
	for (i = 0; i < numOfClusters; i++)
	{
		MPI_Allreduce(&virtualCenters[i].numOfProducts, &allClusters[i].numOfProducts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(virtualCenters[i].virutalCenter.coordinates, allClusters[i].virutalCenter.coordinates, dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		for (j = 0; j < dimensions; j++)
		{
			allClusters[i].virutalCenter.coordinates[j] /= allClusters[i].numOfProducts;
		}
	}
}

Cluster* allocateClustersToGPU(Cluster* allClusters, int numOfClusters)
{
	cudaError_t cudaStatus;
	Cluster* cudaClusters = NULL;
	double* coordsCluster = NULL;
	int i;

	cudaStatus = cudaSetDevice(0);
	handleErrors(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaClusters);

	cudaStatus = cudaMalloc(&cudaClusters, numOfClusters * sizeof(Cluster));
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);

	cudaStatus = cudaMemcpy(cudaClusters, allClusters, numOfClusters * sizeof(Cluster), cudaMemcpyHostToDevice);
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);

#pragma omp parallel for private(coordsCluster)
	for (i = 0; i < numOfClusters; i++)
	{
		cudaStatus = cudaMalloc(&coordsCluster, allClusters[i].virutalCenter.dimensions * sizeof(double));
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);
		cudaStatus = cudaMemcpy(coordsCluster, allClusters[i].virutalCenter.coordinates, allClusters[i].virutalCenter.dimensions * sizeof(double), cudaMemcpyHostToDevice);
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);
		cudaStatus = cudaMemcpy(&cudaClusters[i].virutalCenter.coordinates, &coordsCluster, sizeof(double*), cudaMemcpyHostToDevice);
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);
	}

	return cudaClusters;
}

Product* allocateProductsToGPU(Product* allProducts, int numOfProducts)
{
	cudaError_t cudaStatus;
	Product* cudaProducts = NULL;
	double* coordsProducts = NULL;
	int i;

	cudaStatus = cudaSetDevice(0);
	handleErrors(cudaStatus, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?", cudaProducts);

	cudaStatus = cudaMalloc(&cudaProducts, numOfProducts * sizeof(Product));
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);

	cudaStatus = cudaMemcpy(cudaProducts, allProducts, numOfProducts * sizeof(Product), cudaMemcpyHostToDevice);
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);

#pragma omp parallel for private(coordsProducts)
	for (i = 0; i < numOfProducts; i++)
	{
		cudaStatus = cudaMalloc(&coordsProducts, allProducts[i].dimensions * sizeof(double));
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);
		cudaStatus = cudaMemcpy(coordsProducts, allProducts[i].coordinates, allProducts[i].dimensions * sizeof(double), cudaMemcpyHostToDevice);
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);
		cudaStatus = cudaMemcpy(&cudaProducts[i].coordinates, &coordsProducts, sizeof(double*), cudaMemcpyHostToDevice);
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);
	}
	return cudaProducts;
}

void handleErrors(cudaError_t cudaStatus, const char* errorMessage, void* ptr)
{
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, errorMessage);
		cudaFree(ptr);
		exit(1);
	}
}

void copyProductsFromGPU(Product* allProducts, Product* cudaProducts, int numOfProducts)
{
	cudaError_t cudaStatus;
	double** coordsProducts = (double**)malloc(sizeof(double*)*numOfProducts);
	int i;

#pragma omp parallel for
	for (i = 0; i < numOfProducts; i++)
	{
		coordsProducts[i] = allProducts[i].coordinates;
	}

	cudaStatus = cudaMemcpy(allProducts, cudaProducts, numOfProducts * sizeof(Product), cudaMemcpyDeviceToHost);
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaProducts);

#pragma omp parallel for
	for (i = 0; i < numOfProducts; i++)
	{
		cudaStatus = cudaMemcpy(coordsProducts[i], allProducts[i].coordinates, sizeof(double*), cudaMemcpyDeviceToHost);
		handleErrors(cudaStatus, "cudaMemcpy failed!", cudaProducts);
		cudaFree(allProducts[i].coordinates);
		allProducts[i].coordinates = coordsProducts[i];
	}

	cudaFree(cudaProducts);
	free(coordsProducts);
}

void copyClustersFromGPU(Cluster* allClusters, Cluster* cudaClusters, int numOfClusters)
{
	cudaError_t cudaStatus;
	double** coordsClusters = (double**)malloc(sizeof(double*)*numOfClusters);
	int i;

#pragma omp parallel for
	for (i = 0; i < numOfClusters; i++)
	{
		coordsClusters[i] = allClusters[i].virutalCenter.coordinates;
	}
	cudaStatus = cudaMemcpy(allClusters, cudaClusters, numOfClusters * sizeof(Cluster), cudaMemcpyDeviceToHost);
	handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);

#pragma omp parallel for
	for (i = 0; i < numOfClusters; i++)
	{
		cudaStatus = cudaMemcpy(coordsClusters[i], allClusters[i].virutalCenter.coordinates, sizeof(double*), cudaMemcpyDeviceToHost);
		handleErrors(cudaStatus, "cudaMalloc failed!", cudaClusters);
		cudaFree(allClusters[i].virutalCenter.coordinates);
		allClusters[i].virutalCenter.coordinates = coordsClusters[i];
	}
	free(coordsClusters);
}