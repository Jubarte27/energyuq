#ifndef __driver_h__
#define __driver_h__

#ifdef __cplusplus
extern "C" {
#endif

void DRIVER_Initialize(const int sx, const int sy, const int sz, const int bord,
		       float dx, float dy, float dz, float dt,
		       float *  vpz, float *  vsv, float *  epsilon, float *  delta,
		       float *  phi, float *  theta,
		       float *  pp, float *  pc, float *  qp, float *  qc);

void DRIVER_Finalize();

void DRIVER_Propagate(const int sx, const int sy, const int sz, const int bord,
	       const float dx, const float dy, const float dz, const float dt, const int it, 
	       float * pp, float * pc, float * qp, float * qc);

void DRIVER_Update_pointers(const int sx, const int sy, const int sz, float *pc);

void DRIVER_InsertSource(float dt, int it, int iSource, float *p, float*q, float src);

#ifdef __cplusplus
}
#endif
#endif
