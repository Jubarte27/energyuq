#include "utils.h"
#include "source.h"
#include "driver.h"
#include "fletcher.h"
#include "walltime.h"
#include "model.h"

#define MODEL_GLOBALVARS
#include "precomp.h"
#undef MODEL_GLOBALVARS


void ReportProblemSizeCSV(const int sx, const int sy, const int sz,
			  const int bord, const int st, 
			  FILE *f){
  fprintf(f,
	  "sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",
	  sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,
		      long HWM, char *HWMUnit, FILE *f){
  fprintf(f,
	  "walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",
	  walltime, MSamples, HWM, HWMUnit);
}


void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, 
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, 
	   float *  pp, float *  pc, float *  qp, float *  qc,
	   float *  vpz, float *  vsv, float *  epsilon, float *  delta,
	   float *  phi, float *  theta)
{

  	float tSim=0.0;
  	int nOut=1;
  	float tOut=nOut*dtOutput;

  	const long samplesPropagate=(long)(sx-2*bord)*(long)(sy-2*bord)*(long)(sz-2*bord);
  	const long totalSamples=samplesPropagate*(long)st;

#define MODEL_INITIALIZE
#include "precomp.h"
#undef MODEL_INITIALIZE

#define MEGA 1.0e-6
#define GIGA 1.0e-9
  // DRIVER_Initialize initialize target, allocate data etc
  	DRIVER_Initialize(sx, sy, sz, bord, dx, dy, dz, dt, vpz, vsv, epsilon, delta, phi, theta, pp, pc, qp, qc);
  
  	double walltime=0.0, dumpSlice=0.0;

	double iterTime = wtime();
	double theoretical_fetch_size = 16* ((sx) * (sy) * (sz)) *sizeof(float);
	double theoretical_write_size = 2* (((sx-bord-bord) * (sy-bord-bord) * (sz-bord-bord)) *sizeof(float));
  	for (int it=1; it<=st; it++) {
  		float src = Source(dt, it-1);
      		DRIVER_InsertSource(dt,it-1,iSource,pc,qc,src);
    		const double t0=wtime();
    		DRIVER_Propagate(sx, sy, sz, bord, dx, dy, dz, dt, it, pp, pc, qp, qc);
		SwapArrays(&pp, &pc, &qp, &qc);
    		walltime+=wtime() - t0;
   		tSim=it*dt;
    		if (tSim >= tOut) {
			DRIVER_Update_pointers(sx,sy,sz,pc);
     // 			DumpSliceFile(sx,sy,sz,pc,sPtr);
      			tOut=(++nOut)*dtOutput;
#ifdef _DUMP
		      //	DRIVER_Update_pointers(sx,sy,sz,pc);
			double t1 = wtime();
			DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
			dumpSlice+= wtime() - t1;
#endif
	    	}
  	}

	printf("Total Time iters = %.5f\n", wtime() - iterTime);
	printf("Total Time Dump = %.5f\n", dumpSlice);
  	const char StringHWM[6]="VmHWM";
	char line[256], title[12],HWMUnit[8];
  	long HWM;
  	const double MSamples=(MEGA*(double)totalSamples)/walltime;
	//long double FLOP = 536.0 * (sx-bord - bord) * (sy-bord - bord) * (sz-bord - bord) * st;
 	//printf("FLOPS/s =  %.5f GFLOPS/s\n", ((FLOP/walltime)/GIGA));
  	FILE *fp=fopen("/proc/self/status","r");
  	while (fgets(line, 256, fp) != NULL){
    		if (strncmp(line, StringHWM, 5) == 0) {
      			sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      			break;
    		}
  	}
  	fclose(fp);

  // Dump Execution Metrics
  
  	printf ("Execution time (s) is %lf\n", walltime);
  	printf ("MSamples/s %.0lf\n", MSamples);
  	printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  // Dump Execution Metrics in CSV
  
  	FILE *fr=NULL;
  	const char fName[]="Report.csv";
  	fr=fopen(fName,"w");

  	ReportProblemSizeCSV(sx, sy, sz, bord, st, fr);
  	ReportMetricsCSV(walltime, MSamples, HWM, HWMUnit, fr);
  
	fclose(fr);

	fflush(stdout);

  	DRIVER_Finalize();

}

