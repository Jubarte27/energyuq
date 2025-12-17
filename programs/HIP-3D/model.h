#ifndef _MODEL
#define _MODEL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr, 
           const int sx, const int sy, const int sz, const int bord,
           const float dx, const float dy, const float dz, const float dt, const int it, 
	   float *  pp, float *  pc, float *  qp, float *  qc,
	   float *  vpz, float *  vsv, float *  epsilon, float *  delta,
	   float *  phi, float *  theta);

#endif
