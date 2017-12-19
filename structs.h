#ifndef __STRUCTS_H
#define __STRUCTS_H

#define LEN 20
#define NO_CLUSTER -1

typedef struct
{
	char name[LEN];
	int dimensions;
	int clusterId;
	double* coordinates;

}Product;

typedef struct
{
	int numOfProducts;
	Product virutalCenter;
}Cluster;

#endif