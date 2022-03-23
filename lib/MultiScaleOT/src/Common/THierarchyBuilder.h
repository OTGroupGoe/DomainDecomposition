#ifndef THierarchyBuilder_H_
#define THierarchyBuilder_H_

#include<cstdlib>
#include<cmath>
#include<vector>
#include<Common/THierarchicalPartition.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class THierarchyBuilderNode {
public:
	std::vector<int> children;
	std::vector<int> leaves;
	std::vector<int> posCode;
	int parent;
};

class THierarchyBuilderLayer {
public:
	std::vector<THierarchyBuilderNode> nodes;
};	


class THierarchyBuilder {
public:
	static const int CM_Tree;
	static const int CM_Grid;
	static const int CM_Manual;
	static const double boxTolerance;
	
	double *points;
	int nPoints; // number of points at lowest level
	int dim;
	std::vector<double> boxLo, boxHi; // lower and higher bounds for box around point cloud;
	std::vector<THierarchyBuilderLayer> layers;
	int childMode;
	
	THierarchyBuilder(double *_points, int _nPoints, int _dim,
			int _childMode, int partitionDepth);
	
	void manualAddLayer(int _nNodes, int *_parents, int *_nChildren, int **_children, int *_nLeaves, int **_leaves);
	
	void setBox();
	void setRoot();
	void refine();
	std::vector<std::vector<int> > getChildrenPosCodes(int layerId, int nodeId);
	std::vector<std::vector<int> > getChildrenLeaves(int layerId, int nodeId,
		const std::vector<std::vector<int> >& childrenPosCodes);
	void getRelPosCodeFromIndex(int index, int dim, int *posCode);
	void getOffsetPosCode(int *relPosCode, int *parentPosCode, int dim);
	bool isInBox(double *coord, const int * const posCode, int dim, int layerId);
	void addAtomicLayer();
	
	static double max(double *x, int n, int step, int offset);
	static double min(double *x, int n, int step, int offset);
	
	THierarchicalPartition* convert();
	double** allocateDoubleSignal(int sigdim, int nLayers=0);
	void freeSignal(double **signal, int nLayers);
	void getSignalPos(double **signal);
	int* getDimH(int *finestDims);
	int* getResH();
	
	double** getSignalRadii();
	double** getSignalRadiiAxis(int *axis, int naxis);
	
	void updatePositions(double *newPos);
	void getSignalPosExplicit(double **signal);
	void getSignalRadiiExplicit(double **posH, double **radii);

};

#endif
