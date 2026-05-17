#include <stdio.h>

#define ind(ix,iy,iz) (((iz)*sy+(iy))*sx+(ix))

#define Der2(p, i, s, d2inv) ((K0*p[i]+ K1*(p[i+s]+p[i-s])+ K2*(p[i+2*s]+p[i-2*s]) + K3*(p[i+3*s]+p[i-3*s]) + K4*(p[i+4*s]+p[i-4*s]))*(d2inv))

int main(int argc, char ** argv){

	int sx = 120, sy = 120, sz = 120;
	int strideX=ind(1,0,0)-ind(0,0,0); 
	int strideY=ind(0,1,0)-ind(0,0,0); 
	int strideZ=ind(0,0,1)-ind(0,0,0); 
	for(int iz = 0; iz < sz; iz++){
		for(int iy = 0; iy < sy; iy++){
			for(int ix = 0; ix < sx; ix++){
				int i = ind(ix, iy, iz);
					printf("iz = %d, iy = %d, ix = %d --> %d\n", iz, iy, ix, i);
				printf("Der2X[%d] usa p[%d], p[%d]+p[%d] + p[%d]+p[%d] + p[%d] + p[%d] + p[%d] + p[%d]\n", i, i, i+strideX, i-strideX, i+2*strideX, i-2*strideX, i+3*strideX, i-3*strideX, i+4*strideX, i-4*strideX);
				printf("Der2Y[%d] usa p[%d], p[%d]+p[%d] + p[%d]+p[%d] + p[%d] + p[%d] + p[%d] + p[%d]\n", i, i, i+strideY, i-strideY, i+2*strideY, i-2*strideY, i+3*strideY, i-3*strideY, i+4*strideY, i-4*strideY);
				printf("Der2Z[%d] usa p[%d], p[%d]+p[%d] + p[%d]+p[%d] + p[%d] + p[%d] + p[%d] + p[%d]\n", i, i, i+strideZ, i-strideZ, i+2*strideZ, i-2*strideZ, i+3*strideZ, i-3*strideZ, i+4*strideZ, i-4*strideZ);
			}
		}
	}
	return 0;

}
