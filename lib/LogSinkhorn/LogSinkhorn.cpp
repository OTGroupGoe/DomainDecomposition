#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


//using namespace std;


struct TLogNr {
	double b; // base
	double e; // exponent
};


void addLogNr(TLogNr& a, const TLogNr& b, const double eps) {
	if (a.e>=b.e) {
		a.b+=b.b*std::exp((b.e-a.e)/eps);
	} else {
		a.b*=std::exp((a.e-b.e)/eps);		
		a.b+=b.b;
		a.e=b.e;
	}
}

void addLogNr(TLogNr& a, const double newB, const double newE, const double eps) {
	if (a.e>=newE) {
		a.b+=newB*std::exp((newE-a.e)/eps);
	} else {
		a.b*=std::exp((a.e-newE)/eps);		
		a.b+=newB;
		a.e=newE;
	}
}


void printLogNr(const TLogNr& a) {
	printf("(%e,%e)\n",a.b,a.e);
}

double logsumexp(const double* const c, const double* const beta, const double* const weights, size_t res, double eps) {
	TLogNr result={weights[0],beta[0]-c[0]};
	for(size_t i=1;i<res;i++) {
		addLogNr(result, weights[i], beta[i]-c[i],eps);
	}

	if(!std::isfinite(result.e+eps*std::log(result.b))) {
		printf("\nend:\n");
		printLogNr(result);
	}


	return result.e+eps*std::log(result.b);
}

void iterate(
		double* alpha, double* beta,
		const double* const c, const double* const cT,
		const double* const muX, const double* const muY,
		const double* const rhoX, const double* const rhoY,
		const size_t xres, const size_t yres,
		const double eps, const size_t nIterations) {
	
	for(size_t n=0;n<nIterations;n++) {
		for(size_t x=0;x<xres;x++) {
			alpha[x]=eps*std::log(muX[x]/rhoX[x])-logsumexp(c+x*yres, beta, rhoY, yres, eps);
		}
		for(size_t y=0;y<yres;y++) {
			beta[y]=eps*std::log(muY[y]/rhoY[y])-logsumexp(cT+y*xres, alpha, rhoX, xres, eps);
		}
	}
	
}

double getL1Error(
		double* alpha, double* beta,
		const double* const c,
		const double* const muX,
		const double* const rhoX, const double* const rhoY,
		const size_t xres, const size_t yres,
		const double eps) {

	double result=0;
	// only need to check X-marginal, since Y-marginal is essentially exact after every iteration,
	// since beta update is done after alpha update
	
	for(size_t x=0;x<xres;x++) {
		double conv=logsumexp(c+x*yres, beta, rhoY, yres, eps);
		result+=std::abs(muX[x]-rhoX[x]*std::exp((alpha[x]+conv)/eps));
	}
	return result;

}

int balanceMeasures(double* nu1, double* nu2, const size_t res, const double delta) {
	// nu1, nu2: two non-negative measures of same 1d size
	// res: length of nu1, nu2
	// delta: amount of mass that must be transferred from nu1 to nu2, while
	// keeping both non-negative and while keeping nu1+nu2 unchanged

	// following code assumes that delta>=0
	// so if delta<=0, flip roles of nu1 and nu2
	
	double *mu1,*mu2;
	double eta;
	if(delta>=0) {
		mu1=nu1;
		mu2=nu2;
		eta=delta;
	} else {
		mu1=nu2;
		mu2=nu1;
		eta=-delta;
	}
	
	
	// first try to only transfer mass, where mu2 is already positive (>=thresh) (to not create new support)
	double thresh=1E-10;
	for(size_t i=0;i<2;i++) {
		for(size_t x=0;x<res;x++) {
			if((mu1[x]>0.) && (mu2[x]>=thresh)) {
				if(mu1[x]<eta) {
					mu2[x]+=mu1[x];
					eta-=mu1[x];
					mu1[x]=0.;
				} else {
					mu2[x]+=eta;
					mu1[x]-=eta;
					return 0;
				}
			}
			
		}
		// if fist loop was not successfull / sufficient, set thresh to 0
		thresh=0;
	}
	// if the second run was still not successfull, return failure
	return 1;
	
}

int balanceMeasures_sparse(double* nu1, double* nu2, int *nu1Ind, int *nu2Ind, const size_t nu1Res, const size_t nu2Res, const double delta) {
	// nu1, nu1Ind, res1: sparse representation of 1d non-negative measure
	// nu1: non-zero values
	// nu1Ind: indices of non-zero values
	// nu1Res: number of non-zero values
	// same for nu2, nu2Ind, nu2Res.
	
	// delta: amount of mass that must be transferred from nu1 to nu2, while
	// keeping both non-negative and while keeping nu1+nu2 unchanged
	
	// IMPORTANT: assumes that index lists are sorted
	
	// for now: only transfer mass on common support. do not change support (i.e. sparse data structure) for any of the two measures

	// following code assumes that delta>=0
	// so if delta<=0, flip roles of nu1 and nu2
	
	double *mu1,*mu2;
	int *ind1, *ind2;
	size_t res1, res2;
	double eta;
	if(delta>=0) {
		mu1=nu1;
		ind1=nu1Ind;
		res1=nu1Res;
		
		mu2=nu2;
		ind2=nu2Ind;
		res2=nu2Res;
		
		eta=delta;
	} else {
		mu1=nu2;
		ind1=nu2Ind;
		res1=nu2Res;
		
		mu2=nu1;
		ind2=nu1Ind;
		res2=nu1Res;

		eta=-delta;
	}
	
	
	// first try to only transfer mass, where mu2 is already positive (>=thresh) (to not create new support)
	double thresh=1E-10;
	for(int runNr=0;runNr<2;runNr++) {
		size_t i=0;
		size_t j=0;
		while((i<res1) && (j<res2)) {
			int x=ind1[i];
			int y=ind2[j];
			if(x==y) {
				// if second array has "actual" support in y, or if already in second run
				if((mu2[j]>=thresh) || (runNr>0)) {
					if(mu1[i]<eta) {
						mu2[j]+=mu1[i];
						eta-=mu1[i];
						mu1[i]=0.;
						// this x entry is depleted:
						i++;
					} else {
						mu2[j]+=eta;
						mu1[i]-=eta;
						// complete delta was shifted, return success
						return 0;
					}
				} else {
					j++;
				}
			} else {
				if(x<y) {
					i++;
				} else {
					j++;
				}
			}
		}
	}	
	// if balancing was not successfull, return failure
	return 1;
	
}


////////////////////////////////////////////////////////////////


//struct TDoubleMatrix {
//	double *data;
//	int depth;
//	int *dimensions;
//	bool freeDim;
//	bool freeData;
//	~TDoubleMatrix() {
//		if(freeDim) {
//			free(dimensions);
//		}
//		if(freeData) {
//			free(data);
//		}
//	}
//};


//TDoubleMatrix getMatrixFromNumpyArray(py::array_t<double> mu) {
//	TDoubleMatrix result;
//	
//	py::buffer_info buffer = mu.request();
//	
//	result.data=(double*) buffer.ptr;
//	result.depth=buffer.ndim;
//	result.dimensions=(int*) malloc(sizeof(int)*result.depth);
//	for(int i=0;i<result.depth;i++) {
//		result.dimensions[i]=(int) buffer.shape[i];
//	}
//	result.freeDim=true;
//	result.freeData=false;
//	return result;
//}

template<class T>
T* getDataPointer(py::array_t<T> &mu) {
	return (T*) mu.request().ptr;
}

template<class T>
size_t getDataSize(py::array_t<T> &mu) {
	return mu.request().shape[0];
}


/////////////////////////////////////////////////////////////////////////////////////////
// aux class for adding multiple sparse vectors
// the class starts with an empty list and can then be called with add() to add another sparse vector
// all indices are assumed to be sorted
class TSparseArrayAdder {
public:
	std::vector<int> indices;
	std::vector<double> data;
	size_t size;
	TSparseArrayAdder(): indices(), data(), size(0) {};
	void add(const double* const newData, const int* const newIndices, const size_t n) {
		// i: index in newIndices/newData
		// j: index in indices/data
		size_t i,j;
		// iIndex,jIndex: values of newData, data at i,j
		int iIndex,jIndex;
		i=0;
		j=0;
		if(size>0) {
			jIndex=indices[0];
		}
		
		while((i<n) && (j<size)) {
			iIndex=newIndices[i];
			// increase j, until jIndex>=iIndex, or end of list is reached
			while((jIndex<iIndex) && (j<size)) {
				j++;
				if(j<size) {
					jIndex=indices[j];
				}					
			}
			if(j>=size) {
				break;
			}
			if(jIndex==iIndex) {
				// if iIndex already contained in base list just add value
				data[j]+=newData[i];
			}
			if(jIndex>iIndex) {
				// if jIndex is finally larger, i.e. the value iIndex is still missing, add entry i before j
				data.insert(data.begin()+j,newData[i]);
				indices.insert(indices.begin()+j,newIndices[i]);
				size++;
				j++;
			}
			i++;
		}
		while(i<n) {
			// if i still < n after the above loop, the end of the base list was reached and the remaining entries
			// of the new list just need to be added to the end of the base list
			data.push_back(newData[i]);
			indices.push_back(newIndices[i]);
			size++;
			i++;
		}
	}
};


py::tuple getSparseArrayDataTuple(const TSparseArrayAdder &SparseArrayAdder) {

	py::array_t<double> resultData = py::array_t<double>(SparseArrayAdder.size);
	py::array_t<int> resultIndices = py::array_t<int>(SparseArrayAdder.size);

	std::memcpy(resultData.request().ptr,SparseArrayAdder.data.data(),sizeof(double)*SparseArrayAdder.size);
	std::memcpy(resultIndices.request().ptr,SparseArrayAdder.indices.data(),sizeof(int)*SparseArrayAdder.size);

	return py::make_tuple(resultData,resultIndices);
}


/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////


py::tuple getEuclideanCost(py::array_t<double> &posX, py::array_t<double> &posY) {
	const size_t dim=posX.request().shape[1];
	const size_t xres=posX.request().shape[0];
	const size_t yres=posY.request().shape[0];
	
	const double* const datX= (double*) posX.request().ptr;
	const double* const datY= (double*) posY.request().ptr;

	py::array_t<double> c = py::array_t<double>(xres*yres);
	py::array_t<double> cT = py::array_t<double>(xres*yres);
	double* const datC=(double*) c.request().ptr;
	double* const datCT=(double*) cT.request().ptr;

	for(size_t x=0;x<xres;x++) {
		for(size_t y=0;y<yres;y++) {
			double result=0;
			for(size_t d=0;d<dim;d++) {
				result+=std::pow(datX[x*dim+d]-datY[y*dim+d],2);
			}
			datC[x*yres+y]=result;
			datCT[y*xres+x]=result;
		}
	}

    
	return py::make_tuple(c,cT);
}



/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void PyIterate(
		py::array_t<double> &alpha, py::array_t<double> &beta,
		py::array_t<double> &c, py::array_t<double> &cT,
		py::array_t<double> &muX, py::array_t<double> &muY,
		py::array_t<double> &rhoX, py::array_t<double> &rhoY,
		const double eps, const size_t nIterations) {
	
	double *alphaP=getDataPointer<double>(alpha);
	double *betaP=getDataPointer<double>(beta);
	double *cP=getDataPointer<double>(c);
	double *cTP=getDataPointer<double>(cT);
	double *muXP=getDataPointer<double>(muX);
	double *muYP=getDataPointer<double>(muY);
	double *rhoXP=getDataPointer<double>(rhoX);
	double *rhoYP=getDataPointer<double>(rhoY);

	size_t xres=getDataSize<double>(alpha);
	size_t yres=getDataSize<double>(beta);

	iterate(
			alphaP, betaP,
			cP, cTP,
			muXP, muYP,
			rhoXP, rhoYP,
			xres, yres,
			eps, nIterations
			);
	
	return;
}

int PyIterateUntilError(
		py::array_t<double> &alpha, py::array_t<double> &beta,
		py::array_t<double> &c, py::array_t<double> &cT,
		py::array_t<double> &muX, py::array_t<double> &muY,
		py::array_t<double> &rhoX, py::array_t<double> &rhoY,
		const double eps,
		const size_t nIterationsMax, const size_t nIterationsInner, double maxError) {
	
	double *alphaP=getDataPointer<double>(alpha);
	double *betaP=getDataPointer<double>(beta);
	double *cP=getDataPointer<double>(c);
	double *cTP=getDataPointer<double>(cT);
	double *muXP=getDataPointer<double>(muX);
	double *muYP=getDataPointer<double>(muY);
	double *rhoXP=getDataPointer<double>(rhoX);
	double *rhoYP=getDataPointer<double>(rhoY);

	size_t xres=getDataSize<double>(alpha);
	size_t yres=getDataSize<double>(beta);

	size_t n=0;
	double error=maxError+1.;
	while((n<nIterationsMax) && (error>=maxError)) {
		iterate(
				alphaP, betaP,
				cP, cTP,
				muXP, muYP,
				rhoXP, rhoYP,
				xres, yres,
				eps, nIterationsInner
				);
		n+=nIterationsInner;
		error=getL1Error(
				alphaP, betaP,
				cP,
				muXP,
				rhoXP, rhoYP,
				xres, yres,
				eps
				);
	}
	if(error<maxError) {
		return 0;
	}
	return 1;
}


double PyError(
		py::array_t<double> &alpha, py::array_t<double> &beta,
		py::array_t<double> &c,
		py::array_t<double> &muX,
		py::array_t<double> &rhoX, py::array_t<double> &rhoY,
		const double eps) {
	
	double *alphaP=getDataPointer<double>(alpha);
	double *betaP=getDataPointer<double>(beta);
	double *cP=getDataPointer<double>(c);
	double *muXP=getDataPointer<double>(muX);
	double *rhoXP=getDataPointer<double>(rhoX);
	double *rhoYP=getDataPointer<double>(rhoY);

	size_t xres=getDataSize<double>(alpha);
	size_t yres=getDataSize<double>(beta);

	return getL1Error(
			alphaP, betaP,
			cP,
			muXP,
			rhoXP, rhoYP,
			xres, yres,
			eps
			);
	
}

int PyBalanceMeasures(
		py::array_t<double> &nu1, py::array_t<double> &nu2,
		const double delta) {
	
	double *nu1P=getDataPointer<double>(nu1);
	double *nu2P=getDataPointer<double>(nu2);

	size_t res=getDataSize<double>(nu1);
	return balanceMeasures(nu1P, nu2P, res, delta);
}


int PyBalanceMeasures_Sparse(
		py::array_t<double> &nu1, py::array_t<double> &nu2,
		py::array_t<int> &nu1Ind, py::array_t<int> &nu2Ind,
		
		const double delta) {
	
	double *nu1P=getDataPointer<double>(nu1);
	double *nu2P=getDataPointer<double>(nu2);

	int *nu1IP=getDataPointer<int>(nu1Ind);
	int *nu2IP=getDataPointer<int>(nu2Ind);


	size_t nu1Res=getDataSize<double>(nu1);
	size_t nu2Res=getDataSize<double>(nu2);

	return balanceMeasures_sparse(nu1P, nu2P, nu1IP, nu2IP, nu1Res, nu2Res, delta);
}




PYBIND11_MODULE(LogSinkhorn, m) {
	m.doc() = "Simple LogSinkhorn implementation with running max"; // module docstring
	m.def("iterate", &PyIterate, "Do LogSinkhorn iterations.",
		py::arg("alpha"), py::arg("beta"),
		py::arg("c"), py::arg("cT"),
		py::arg("muX"), py::arg("muY"),
		py::arg("rhoX"), py::arg("rhoY"),
		py::arg("eps"), py::arg("nIterations")
		);

	m.def("iterateUntilError", &PyIterateUntilError, "Do LogSinkhorn iterations until L1 error is sufficiently small or maximal number of iterations is reached.",
		py::arg("alpha"), py::arg("beta"),
		py::arg("c"), py::arg("cT"),
		py::arg("muX"), py::arg("muY"),
		py::arg("rhoX"), py::arg("rhoY"),
		py::arg("eps"),
		py::arg("nIterationsMax"), py::arg("nIterationsInner"), py::arg("maxError")
		);

	m.def("L1error", &PyError, "Compute L1 error of X-marginal",
		py::arg("alpha"), py::arg("beta"),
		py::arg("c"),
		py::arg("muX"),
		py::arg("rhoX"), py::arg("rhoY"),
		py::arg("eps")
		);

	m.def("balanceMeasures", &PyBalanceMeasures, "Balance two measures while preserving positivity and their sum.",
		py::arg("nu1"), py::arg("nu2"),
		py::arg("delta")
		);
	
	m.def("balanceMeasures_sparse", &PyBalanceMeasures_Sparse, "Balance two measures while preserving positivity and their sum.",
		py::arg("nu1"), py::arg("nu2"),
		py::arg("nu1Ind"), py::arg("nu2Ind"),
		py::arg("delta")
		);

	py::class_<TSparseArrayAdder>(m, "TSparseArrayAdder")
	.def(py::init<>())
	.def("add", [](TSparseArrayAdder &SparseArrayAdder, py::array_t<double> &data, py::array_t<int> &indices) {
		double *dataP=getDataPointer<double>(data);
		int *indicesP=getDataPointer<int>(indices);
		size_t size=getDataSize<double>(data);
		SparseArrayAdder.add(dataP, indicesP, size);
        })
        .def("getDataTuple",[](const TSparseArrayAdder &SparseArrayAdder) {
            return getSparseArrayDataTuple(SparseArrayAdder);
        });

	m.def("getEuclideanCost", &getEuclideanCost);

}

