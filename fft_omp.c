/******************************************************************************/
/* 
  Author:
    Original C version by Wesley Petersen.
    This C version by John Burkardt.
  Reference:
    Wesley Petersen, Peter Arbenz, 
    Introduction to Parallel Computing - A practical guide with examples in C,
    Oxford University Press,
    ISBN: 0-19-851576-6,
    LC: QA76.58.P47.
*/

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <float.h>
# include <omp.h>

// #define MAX 26 // standard
#define MAX 19 // small

const double PI = 3.141592653589793;

void cffti(int n, double w[]){
    const double aw = 2.0 * PI / ( ( double ) n );
    const int n2 = n / 2;
    int i = 0;
    #pragma omp parallel for private(i)
    for (i = 0; i < n2; i++){
        double temp;
        double arg = aw * ((double ) i);
        w[i*2+0] = cos (arg);
        // temp = cos(arg);
        w[i*2+1] = sin (arg);
	// w[i*2+0] = sin(arg);
        // temp = sin(arg);
    }
    return;
}

int main (int argc, char **argv){;
    double *w;

    printf ( "FFT_SERIAL C version\n" );
    fflush(stdout);

    int n = 1;
    double t1 = omp_get_wtime();
    for (int ln2 = 1; ln2 <= MAX; ln2++ ){
        n = n << 1;
        w = (double*) malloc(n*sizeof(double));
        for (int icase = 0; icase < 2; icase++){
            cffti ( n, w );
        }
        free ( w );
    }
    double t2 = omp_get_wtime() - t1;
    printf("Execution Time: %.15f\n", t2);
    return 0;
}
