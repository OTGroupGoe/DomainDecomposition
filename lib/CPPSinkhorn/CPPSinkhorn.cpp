#include<cstdlib>
#include<cstdio>

#include<Common.h>
#include<Sinkhorn.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


using namespace std;

TDoubleMatrix getMatrixFromNumpyArray(py::array_t<double> mu) {
	TDoubleMatrix result;
	
	py::buffer_info buffer = mu.request();
	
	result.data=(double*) buffer.ptr;
	result.depth=buffer.ndim;
	result.dimensions=(int*) malloc(sizeof(int)*result.depth);
	for(int i=0;i<result.depth;i++) {
		result.dimensions[i]=(int) buffer.shape[i];
	}
	return result;
}


int SolveSinkhorn(TDoubleMatrix *muX, TDoubleMatrix *muY, TDoubleMatrix *posX, TDoubleMatrix *posY,
		TDoubleMatrix *rhoX, TDoubleMatrix *rhoY,
		TDoubleMatrix *alphaInit,
		double SinkhornError, double eps,
		bool doubleCTransform,
		TDoubleMatrix *resultAlpha, TDoubleMatrix *resultBeta,
		TSparseCSRContainer *resultKernelData,
		int test) {

	
	// fundamental parameters
	int depth=0;
	int layerFinest=depth; // what is the finest layer we want to solve?
	int layerCoarsest=depth; // coarsest layer to solve on.
	int dim=posX->dimensions[1]; // spatial dimension

	int msg; // store return codes from functions


	///////////////////////////////////////////////
	// problem setup
	TMultiScaleSetup MultiScaleSetupX(posX->data, muX->data, posX->dimensions[0],posX->dimensions[1], depth,THierarchyBuilder::CM_Tree,true,true,false);


	TMultiScaleSetup MultiScaleSetupY(posY->data, muY->data, posY->dimensions[0],posY->dimensions[1], depth,THierarchyBuilder::CM_Tree,true,true,false);

	double **rhoXH=MultiScaleSetupX.HB->allocateDoubleSignal(1);
	MultiScaleSetupX.HP->computeHierarchicalMasses(rhoX->data,rhoXH);
	double **rhoYH=MultiScaleSetupY.HB->allocateDoubleSignal(1);
	MultiScaleSetupY.HP->computeHierarchicalMasses(rhoY->data,rhoYH);
	
		
	/////////////////////////////////////////////////////////////////////////////////////////
	// setup a cost function provider
	THierarchicalCostFunctionProvider_SquaredEuclidean costProvider(
			MultiScaleSetupX.posH, MultiScaleSetupY.posH,
			NULL, NULL,
			dim, layerFinest,
			true,
			MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH,
			1.
			);

	int xres=muX->dimensions[0];
	int yres=muY->dimensions[0];

	std::memcpy(MultiScaleSetupX.alphaH[layerFinest],alphaInit->data,sizeof(double)*xres);

		
	MultiScaleSetupX.HP->signal_propagate_double(MultiScaleSetupX.alphaH, 0, layerFinest, THierarchicalPartition::MODE_MAX);
	// compute beta via c-transform
	THierarchicalDualMaximizer::getMaxDual(MultiScaleSetupX.HP, MultiScaleSetupY.HP,
			MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH, layerFinest,
			&costProvider,
			THierarchicalDualMaximizer::MODE_BETA);	
	MultiScaleSetupY.HP->signal_propagate_double(MultiScaleSetupY.alphaH, 0, layerFinest, THierarchicalPartition::MODE_MAX);
	
	if (doubleCTransform) {
		// do another c-transform on alpha
		THierarchicalDualMaximizer::getMaxDual(MultiScaleSetupX.HP, MultiScaleSetupY.HP,
				MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH, layerFinest,
				&costProvider,
				THierarchicalDualMaximizer::MODE_ALPHA);	
		MultiScaleSetupX.HP->signal_propagate_double(MultiScaleSetupX.alphaH, 0, layerFinest, THierarchicalPartition::MODE_MAX);
	}	
	/////////////////////////////////////////////////////////////////////////////////////////
	// epsScaling	
	TEpsScalingHandler epsScalingHandler;
	epsScalingHandler.setupGeometricSingleLayer(depth+1,eps,eps,0); // single eps value on single layer

	
	TSinkhornSolverBase::TSinkhornSolverParameters cfg={
			100000, // maxIterations
			100, // innerIterations
			1000, // maxAbsorptionLoops
			1E5, // absorption_scalingBound
			1E100, // absorption_scalingLowerBound
			1E-10, // truncation_thresh
			true, // refineKernel when refining layer (as opposed to attempting to estimate directly)
			true // automatically recompute kernel when problem is solved
			};

	/////////////////////////////////////////////////////////////////////////////////////////
	// create solver object
	TSinkhornSolverStandard SinkhornSolver(MultiScaleSetupX.nLayers,
			epsScalingHandler.nEpsLists, epsScalingHandler.epsLists,
			layerCoarsest, layerFinest, SinkhornError,
			cfg,
			MultiScaleSetupX.HP, MultiScaleSetupY.HP,
			MultiScaleSetupX.muH, MultiScaleSetupY.muH,
			rhoXH, rhoYH,
			MultiScaleSetupX.alphaH, MultiScaleSetupY.alphaH,
			&costProvider
			);


	

	SinkhornSolver.initialize();
	SinkhornSolver.kernelGenerator.useSafeMode=false;
	SinkhornSolver.kernelGenerator.useFixDuals=false;
	
	if(test==0) {
		msg=SinkhornSolver.solve();	
		//printf("return code: %d\n",msg);
	} else {
		SinkhornSolver.changeLayer(layerFinest);
		SinkhornSolver.changeEps(eps);

		msg=0;
	}
	
	std::memcpy(resultAlpha->data,MultiScaleSetupX.alphaH[layerFinest],sizeof(double)*xres);
	std::memcpy(resultBeta->data,MultiScaleSetupY.alphaH[layerFinest],sizeof(double)*yres);
	*resultKernelData=SinkhornKernelGetCSRData(SinkhornSolver.kernel);
	//SinkhornSolver.writeMarginalY(muYResult->data);

	MultiScaleSetupX.HB->freeSignal(rhoXH,MultiScaleSetupX.HP->nLayers);
	MultiScaleSetupY.HB->freeSignal(rhoYH,MultiScaleSetupY.HP->nLayers);
	
	return msg;
}


py::tuple PySolveSinkhorn(py::array_t<double> muX, py::array_t<double> muY,
		py::array_t<double> posX, py::array_t<double> posY,
		py::array_t<double> rhoX, py::array_t<double> rhoY,
		py::array_t<double> alphaInit,
		double SinkhornError, double eps,
		bool doubleCTransform,		
		int test) {
	
	TDoubleMatrix muXMat=getMatrixFromNumpyArray(muX);
	TDoubleMatrix muYMat=getMatrixFromNumpyArray(muY);
	TDoubleMatrix posXMat=getMatrixFromNumpyArray(posX);
	TDoubleMatrix posYMat=getMatrixFromNumpyArray(posY);
	TDoubleMatrix rhoXMat=getMatrixFromNumpyArray(rhoX);
	TDoubleMatrix rhoYMat=getMatrixFromNumpyArray(rhoY);
	TDoubleMatrix alphaInitMat=getMatrixFromNumpyArray(alphaInit);



	/* allocate the output arrays */
	py::array_t<double> resultAlpha = py::array_t<double>(alphaInit.request().size);
	TDoubleMatrix resultAlphaMat=getMatrixFromNumpyArray(resultAlpha);
	
	py::array_t<double> resultBeta = py::array_t<double>(muY.request().size);
	TDoubleMatrix resultBetaMat=getMatrixFromNumpyArray(resultBeta);

	TSparseCSRContainer resultKernelData;

	int msg;
	msg=SolveSinkhorn(&muXMat, &muYMat, &posXMat, &posYMat, &rhoXMat, &rhoYMat,
			&alphaInitMat,
			SinkhornError, eps,
			doubleCTransform,
			&resultAlphaMat, &resultBetaMat,
			&resultKernelData,test);

	free(muXMat.dimensions);
	free(muYMat.dimensions);
	free(posXMat.dimensions);
	free(posYMat.dimensions);
	free(alphaInitMat.dimensions);
	free(resultAlphaMat.dimensions);
	free(resultBetaMat.dimensions);

	
	/*if(msg!=0) {
		return py::make_tuple(msg);
	}*/
	
	
	//printf("done.\n");
	//printf("kernel entries: %d\n",resultKernelData.nonZeros);

	/* copy kernel data */
	py::array_t<double> resultKernelDataData = py::array_t<double>(resultKernelData.nonZeros);
	py::array_t<int> resultKernelDataPos = py::array_t<int>(resultKernelData.nonZeros);
	py::array_t<int> resultKernelDataIndptr = py::array_t<int>(resultKernelData.xres+1);
	std::memcpy(resultKernelDataData.request().ptr,resultKernelData.data.data(),sizeof(double)*resultKernelData.nonZeros);
	std::memcpy(resultKernelDataPos.request().ptr,resultKernelData.indices.data(),sizeof(int)*resultKernelData.nonZeros);
	std::memcpy(resultKernelDataIndptr.request().ptr,resultKernelData.indptr.data(),sizeof(int)*(resultKernelData.xres+1));
	



	return py::make_tuple(msg,resultAlpha,resultBeta,resultKernelDataData,resultKernelDataPos,resultKernelDataIndptr);
	

}

PYBIND11_MODULE(CPPSinkhorn, m) {
    m.doc() = "pybind11 interface to OTToolbox Sinkhorn Solver"; // module docstring
    m.def("SolveSinkhorn", &PySolveSinkhorn, "Apply Sinkhorn solver.",
    		py::arg("muX"), py::arg("muY"),
    		py::arg("posX"), py::arg("posY"),
    		py::arg("rhoX"), py::arg("rhoY"),
    		py::arg("alphaInit"), py::arg("SinkhornError"), py::arg("eps"),
    		py::arg("doubleCTransform"),
    		py::arg("test")
    		);
    //m.def("getMatrixFromNumpyArray", &getMatrixFromNumpyArray, "", py::arg("mu"));

 
    
}
