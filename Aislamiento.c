#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
/*Esto puede petar dependiendo del pc y el compilador que uses*/
/*Despues de clases lo tendre que mirar a detalle el codigo el finde*/
#ifdef _WIN32
#include <windows.h> 
static double now_ms(void){
	LARGE_INTEGER freq,count;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&count);
	return (double)count.QuadPart*1000.0/freq.QuadPart;
}
#endif

#define N_BINS 256
#define N_ITERS 500

static double bench(int n_samples,int n_features,int n_threads){
	unsigned char *X=malloc((size_t)n_samples*n_features);
	float *grad=malloc((size_t)n_samples*sizeof(float));
	float *hess=malloc((size_t)n_samples*sizeof(float));
	double *hist=calloc((size_t)n_features*N_BINS*2,sizeof(double));

	if(!X||!grad||!hess||!hist)exit(1);

	int i;
	for(i=0;i<n_samples*n_features;i++)
		X[i]=i%N_BINS;

	for(i=0;i<n_samples;i++){
		grad[i]=(float)(i&7);
		hess[i]=1.0f;
	}

	omp_set_num_threads(n_threads);

	double t0=now_ms();
	int it;

	for(it=0;it<N_ITERS;it++){
		#pragma omp parallel
		{
			double *local=calloc((size_t)n_features*N_BINS*2,sizeof(double));
			int s,f;

			#pragma omp for
			for(s=0;s<n_samples;s++){
				for(f=0;f<n_features;f++){
					int bin=X[s*n_features+f];

					local[(f*N_BINS+bin)*2+0]+=grad[s];
					local[(f*N_BINS+bin)*2+1]+=hess[s];
				}
			}
			#pragma omp critical
			{
				for(i=0;i<n_features*N_BINS*2;i++)
					hist[i]+=local[i];
			}
			free(local);
		}
	}
	double ms=(now_ms()-t0)/N_ITERS;

	free(X);
	free(grad);
	free(hess);
	free(hist);
	return ms;
}
int main(void){
	int threads=omp_get_max_threads();
	int N=20000,F=100;
	printf("Max threads available: %d\n",threads);
	double t1=bench(N,F,1);
	double t2=bench(N,F,threads);
	printf("Samples=%d Features=%d Threads=%d\n",N,F,threads);
	printf("Sequential: %.3f ms\n",t1);
	printf("Parallel  : %.3f ms (%.2fx)\n",t2,t2/t1);

	return 0;
}
