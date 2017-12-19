#pragma warning( disable : 4996 )

#include "kernel.h"
#include "kmeans.h"

//void printClusters(const Cluster* allClusters, int numOfClusters, int numOfDimensions, double quality);
void writeResultsToFile(Cluster* allClusters, int numOfClusters, int dimensions, double qm);
void freeMemory(Product* myProducts, int myProductsSize);

void main(int argc, char* argv[])
{
	int numOfProducts, numOfDimensions, numOfIterations = 0, maxNumOfClusters, maxNumOfIterations = 0,
		numOfClusters = 2, numprocs, myid, myProductsSize, finish = 1;
	double wantedQualityOfClusters, currentQuality, begin, end;
	Product* allProducts = NULL, *myProducts = NULL;
	Cluster* allClusters = NULL;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	MPI_Datatype MPI_Product = makeMPIPoduct();
	MPI_Datatype MPI_Cluster = makeMPICluster(MPI_Product);

	fflush(stdout);
	//read the data from the file
	if (myid == 0)
	{
		begin = MPI_Wtime();
		allProducts = readDataFromFile(&numOfProducts, &numOfDimensions, &maxNumOfClusters, &maxNumOfIterations, &wantedQualityOfClusters);
	}

	//give all processes the variables data
	broadCastAllParameters(&numOfProducts, &maxNumOfIterations, &numOfDimensions, &maxNumOfClusters);
	//give each product a portion of points to work on
	giveProductToProcs(allProducts, numOfProducts, &myProducts, &myProductsSize, numprocs, myid, MPI_Product);

	do
	{

		free(allClusters);
		
		//allocate memory for cluster in the num of clusters size
		allClusters = (Cluster*)calloc(sizeof(Cluster), numOfClusters);

		//start kmeans algorithm
		kmeans(allClusters, numOfClusters, allProducts, numOfProducts, myProducts, myProductsSize, maxNumOfIterations, myid, numprocs, MPI_Cluster, MPI_Product);
	
		if (myid == 0)
		{
			//evaluate the quality of the clusters found (by equation given in word document)
			currentQuality = evaluateQualityOfClusters(allClusters, numOfClusters, allProducts, numOfProducts);

			if (currentQuality < wantedQualityOfClusters)
				finish = 0;
		}
		// 1. stop if the quality is good
		MPI_Bcast(&finish, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// 2. send num of clusters++
		numOfClusters++;
	
	} while (numOfClusters <= maxNumOfClusters && finish);

	if (myid == 0)
	{
		end = MPI_Wtime();
		//printClusters(allClusters, numOfClusters - 1, numOfDimensions, currentQuality);
		writeResultsToFile(allClusters, numOfClusters - 1, numOfDimensions, currentQuality);
		freeMemory(allProducts, numOfProducts);
		free(myProducts);
		printf("\n\n\nnumprocs = %d time = %lf", numprocs, end - begin);
		fflush(stdout);
	}

	free(allClusters);
	if (myid != 0)
		freeMemory(myProducts, myProductsSize);
	MPI_Type_free(&MPI_Product);
	MPI_Type_free(&MPI_Cluster);
	MPI_Finalize();
}

/*
void printClusters(const Cluster* allClusters, int numOfClusters, int numOfDimensions, double quality)
{
	int i, j;

	printf("Number of clusters with best measure\nK = %d QM = %.2lf\nCenters of clusters:\n", numOfClusters, quality);
	for (i = 0; i < numOfClusters; i++)
	{
		printf("C%d: \n", i + 1);
		for (j = 0; j < numOfDimensions; j++)
		{
			printf("%.2lf ", allClusters[i].virutalCenter.coordinates[j]);
			fflush(stdout);
		}
		puts("");
	}
}
*/

void writeResultsToFile(Cluster* allClusters, int numOfClusters, int dimensions, double qm)
{
	
	int i, j;
	FILE* resultFile = fopen("C:\\Users\\afeka\\Desktop\\parallel_computing\\parallel_computing\\result.txt", "w");
	fprintf(resultFile, "Number of clusters with the best measure:\nK = %d QM = %lf\n\nCenters Of Clusters:\n\n", numOfClusters, qm);

	for (i = 0; i < numOfClusters; i++)
	{

		fprintf(resultFile, "C%d  \n", (i + 1));

		for (j = 0; j < dimensions; j++)
		{
			fprintf(resultFile, "%4.3lf    ", allClusters[i].virutalCenter.coordinates[j]);
		}
		fprintf(resultFile, "\n\n");
	}
	fclose(resultFile);
}
void freeMemory(Product* myProducts, int myProductsSize)
{
	int i;
#pragma omp parallel for
	for (i = 0; i < myProductsSize; i++)
	{
		free(myProducts[i].coordinates);
	}
	free(myProducts);
}
