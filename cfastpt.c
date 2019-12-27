#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>

#include <time.h>

#include <fftw3.h>

#include "utils.h"
#include "utils_complex.h"
#include "cfastpt.h"


void fastpt_scalar(int *alpha_ar, int *beta_ar, int *ell_ar, int *isP13type_ar, double *coeff_A_ar, int Nterms, double *Pout, double *k, double *Pin, int Nk){
	long i, j;
	int alpha, beta, ell, isP13type, sign;

	double **Fy;
	Fy = malloc(sizeof(double*) * Nterms);
	for(i=0;i<Nterms;i++) {Fy[i] = malloc(sizeof(double) * Nk);}
	// printf("Nk:%ld\n", Nk);
	fastpt_config config;
	config.nu = -2.; config.c_window_width = 0.25; config.N_pad = 1500;
	config.N_extrap_low = 500; config.N_extrap_high = 500;

	// printf("Nk:%ld\n", Nk_extend);
	// exit(0);
	for(j=0; j<Nk; j++) {Pout[j] = 0.;}
	J_abl_ar(k, Pin, Nk, alpha_ar, beta_ar, ell_ar, isP13type_ar, Nterms, &config, Fy);
	for(i=0; i<Nterms; i++) {
		for(j=0; j<Nk; j++){
			Pout[j] += coeff_A_ar[i] * Fy[i][j];
		}
	}
	for(i = 0; i < Nterms; i++) {free(Fy[i]);}
	free(Fy);
}

void J_abl_ar(double *x, double *fx, long N, int *alpha, int *beta, int *ell, int *isP13type, int Nterms, fastpt_config *config, double **Fy) {
	// x: k array, fx: Pin array

	long N_original = N;
	long N_pad = config->N_pad;
	long N_extrap_low = config->N_extrap_low;
	long N_extrap_high = config->N_extrap_high;
	N += (2*N_pad + N_extrap_low+N_extrap_high);

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	double xi;
	int sign;
	if(N_extrap_low) {
		if(fx[0]==0) {
			printf("Can't log-extrapolate zero on the low side!\n");
			exit(1);
		}
		else if(fx[0]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[1]/fx[0]<=0) {printf("Log-extrapolation on the low side fails due to sign change!\n"); exit(1);}
		double dlnf_low = log(fx[1]/fx[0]);
		for(i=N_pad; i<N_pad+N_extrap_low; i++) {
			xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
			fb[i] = sign * exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
		}
	}
	for(i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) {
		fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
	}
	if(N_extrap_high) {
		if(fx[N_original-1]==0) {
			printf("Can't log-extrapolate zero on the high side!\n");
			exit(1);
		}
		else if(fx[N_original-1]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[N_original-1]/fx[N_original-2]<=0) {printf("Log-extrapolation on the high side fails due to sign change!\n"); exit(1);}
		double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
		for(i=N-N_pad-N_extrap_high; i<N-N_pad; i++) {
			xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
			fb[i] = sign * exp(log(fx[N_original-1]*sign) + (i- N_pad - N_extrap_low- N_original)*dlnf_high) / pow(xi, config->nu);
		}
	}

	fftw_complex *out, *out_vary;
	fftw_complex *out_pad1, *out_pad2;
	fftw_complex *pads_convolve;

	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );

	out_pad1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	out_pad2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	pads_convolve = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (2*N+1) );

	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * (2*N) );
	// double *out_ifft2;
	// out_ifft2 = malloc(sizeof(double) * N );

	plan_backward = fftw_plan_dft_c2r_1d(2*N, out_vary, out_ifft, FFTW_ESTIMATE);

	int i_term;
	double complex h_part[N+1];
	double tau_l[N+1];
	for(i=0;i<=N;i++){
		tau_l[i] = 2.*M_PI / dlnx / N * i;
	}
	double complex fz[N+1];
	double p;
	int sign_ell;

	fftw_complex *a, *b;
	fftw_complex *a1, *b1;
	fftw_complex *c;
	fftw_plan pa, pb, pc;

	long Ntotal_convolve;
	if(N%2==0) { // N+1 is odd
		Ntotal_convolve = 2*N + 1;
	}else {
		printf("This fftconvolve doesn't support even size input arrays (of out_pad1, outpad2)\n"); exit(1);
	}

	// initialize FFT plans for Convolution
	// avoid initialization for each i_term
	a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	a1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	b1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	pa = fftw_plan_dft_1d(Ntotal_convolve, a, a1, FFTW_FORWARD, FFTW_ESTIMATE);
	pb = fftw_plan_dft_1d(Ntotal_convolve, b, b1, FFTW_FORWARD, FFTW_ESTIMATE);
	pc = fftw_plan_dft_1d(Ntotal_convolve, a1, c, FFTW_BACKWARD, FFTW_ESTIMATE);
	/////

	for(i_term=0;i_term<Nterms;i_term++){
		g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + alpha[i_term], eta_m, gl, halfN+1);

		// Do convolutions
		for(i=0; i<=halfN; i++) {
			out_pad1[i+halfN] = out[i] / (double)N * gl[i] ;
			// printf("gl:%lg, %lg\n", creal(gl[i]), cimag(gl[i]));
		}
		for(i=0; i<halfN; i++) {
			out_pad1[i] = conj(out_pad1[N-i]) ;
			// printf("gl:%e\n", gl[i]);
		}

		if(alpha!=beta){
			g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + beta[i_term], eta_m, gl, halfN+1);

			for(i=0; i<=halfN; i++) {
				out_pad2[i+halfN] = out[i] / (double)N * gl[i] ;
				// printf("gl:%e\n", gl[i]);
			}
			for(i=0; i<halfN; i++) {
				out_pad2[i] = conj(out_pad2[N-i]);
				// printf("gl:%e\n", gl[i]);
			}
			// fftconvolve(out_pad1, out_pad2, N+1, pads_convolve);
			fftconvolve_optimize(out_pad1, out_pad2, N+1, pads_convolve, a, b, a1, b1, c, pa, pb, pc);
		}else{
			// fftconvolve(out_pad1, out_pad1, N+1, pads_convolve);
			fftconvolve_optimize(out_pad1, out_pad1, N+1, pads_convolve, a, b, a1, b1, c, pa, pb, pc);
		}
		// convolution finished
		pads_convolve[N] = creal(pads_convolve[N]);

		for(i=0;i<=N;i++){
			h_part[i] = pads_convolve[i+N]; //C_h term in Eq.(2.21) in McEwen et al (2016)
											// but only take h = 0,1,2,...,N.
		}

		p = -5.-2.*config->nu - alpha[i_term]-beta[i_term];
		f_z(p+1, tau_l, fz, N+1);

		for(i=0; i<=N; i++){
			out_vary[i] = h_part[i] * conj(fz[i]) * cpow(2., I*tau_l[i]);
		}
		fftw_execute(plan_backward);

		sign_ell = (ell[i_term]%2? -1:1);
		for(i=0; i<N_original; i++){
			Fy[i_term][i] = out_ifft[2*(i+N_pad+N_extrap_low)] * sign_ell / (M_PI*M_PI) * pow(2., 2.+2*config->nu+alpha[i_term]+beta[i_term]) * pow(x[i],-p-2.);
		}
	}


	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
	free(fb);

	fftw_destroy_plan(pa);
	fftw_destroy_plan(pb);
	fftw_destroy_plan(pc);
	fftw_free(a);
	fftw_free(b);
	fftw_free(a1);
	fftw_free(b1);
	fftw_free(c);	
}

void fastpt_tensor(int *alpha_ar, int *beta_ar, int *J1_ar, int *J2_ar, int *Jk_ar, double *coeff_AB_ar, int Nterms, double *Pout, double *k, double *Pin, int Nk){
	long i, j;
	// int alpha, beta, l1, l2, l;

	double **Fy;
	Fy = malloc(sizeof(double*) * Nterms);
	for(i=0;i<Nterms;i++) {Fy[i] = malloc(sizeof(double) * Nk);}
	// printf("Nk:%ld\n", Nk);
	fastpt_config config;
	config.c_window_width = 0.25; config.N_pad = 1500;
	config.N_extrap_low = 500; config.N_extrap_high = 500;

	// printf("Nk:%ld\n", Nk_extend);
	// exit(0);
	for(j=0; j<Nk; j++) {Pout[j] = 0.;}
	J_abJ1J2Jk_ar(k, Pin, Nk, alpha_ar, beta_ar, J1_ar, J2_ar, Jk_ar, Nterms, &config, Fy);
	for(i=0; i<Nterms; i++) {
		for(j=0; j<Nk; j++){
			Pout[j] += coeff_AB_ar[i] * Fy[i][j];
		}
	}
	for(i = 0; i < Nterms; i++) {free(Fy[i]);}
	free(Fy);
}

void J_abJ1J2Jk_ar(double *x, double *fx, long N, int *alpha, int *beta, int *J1, int *J2, int *Jk, int Nterms, fastpt_config *config, double **Fy) {
	// x: k array, fx: Pin array

	double nu1, nu2;

	long N_original = N;
	long N_pad = config->N_pad;
	long N_extrap_low = config->N_extrap_low;
	long N_extrap_high = config->N_extrap_high;
	N += (2*N_pad + N_extrap_low+N_extrap_high);

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	double f_unbias[N], x_full[N];
	// biased input func
	double *fb1,*fb2;
	fb1 = malloc(N* sizeof(double));
	fb2 = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		x_full[i] = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
		x_full[N-1-i] = exp(log(x0) + (N-1-i-N_pad - N_extrap_low)*dlnx);
		f_unbias[i] = 0.;
		f_unbias[N-1-i] = 0.;
	}
	double xi;
	int sign;
	if(N_extrap_low) {
		if(fx[0]==0) {
			printf("Can't log-extrapolate zero on the low side!\n");
			exit(1);
		}
		else if(fx[0]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[1]/fx[0]<=0) {printf("Log-extrapolation on the low side fails due to sign change!\n"); exit(1);}
		double dlnf_low = log(fx[1]/fx[0]);
		for(i=N_pad; i<N_pad+N_extrap_low; i++) {
			x_full[i] = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
			f_unbias[i] = sign * exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low);
		}
	}
	for(i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) {
		x_full[i] = x[i-N_pad-N_extrap_low];
		f_unbias[i] = fx[i-N_pad-N_extrap_low];
	}
	if(N_extrap_high) {
		if(fx[N_original-1]==0) {
			printf("Can't log-extrapolate zero on the high side!\n");
			exit(1);
		}
		else if(fx[N_original-1]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[N_original-1]/fx[N_original-2]<=0) {printf("Log-extrapolation on the high side fails due to sign change!\n"); exit(1);}
		double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
		for(i=N-N_pad-N_extrap_high; i<N-N_pad; i++) {
			x_full[i] = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
			f_unbias[i] = sign * exp(log(fx[N_original-1]*sign) + (i- N_pad - N_extrap_low- N_original)*dlnf_high);
		}
	}

	fftw_complex *out, *out_vary;
	fftw_complex *out2;
	fftw_complex *out_pad1, *out_pad2;
	fftw_complex *pads_convolve;

	fftw_plan plan_forward, plan_backward;
	fftw_plan plan_forward2;

	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );

	out_pad1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	out_pad2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	pads_convolve = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (2*N+1) );

	plan_forward = fftw_plan_dft_r2c_1d(N, fb1, out, FFTW_ESTIMATE);
	plan_forward2 = fftw_plan_dft_r2c_1d(N, fb2, out2, FFTW_ESTIMATE);
	
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * (2*N) );
	// double *out_ifft2;
	// out_ifft2 = malloc(sizeof(double) * N );

	plan_backward = fftw_plan_dft_c2r_1d(2*N, out_vary, out_ifft, FFTW_ESTIMATE);

	int i_term;
	double complex h_part[N+1];
	double tau_l[N+1];
	for(i=0;i<=N;i++){
		tau_l[i] = 2.*M_PI / dlnx / N * i; // add minus sign convenient for getting fz from g_m_vals
	}
	double complex fz[N+1];
	double p;
	int sign_ell;

	fftw_complex *a, *b;
	fftw_complex *a1, *b1;
	fftw_complex *c;
	fftw_plan pa, pb, pc;

	long Ntotal_convolve;
	if(N%2==0) { // N+1 is odd
		Ntotal_convolve = 2*N + 1;
	}else {
		printf("This fftconvolve doesn't support even size input arrays (of out_pad1, outpad2)\n"); exit(1);
	}

	// initialize FFT plans for Convolution
	// avoid initialization for each i_term
	a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	a1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	b1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
	pa = fftw_plan_dft_1d(Ntotal_convolve, a, a1, FFTW_FORWARD, FFTW_ESTIMATE);
	pb = fftw_plan_dft_1d(Ntotal_convolve, b, b1, FFTW_FORWARD, FFTW_ESTIMATE);
	pc = fftw_plan_dft_1d(Ntotal_convolve, a1, c, FFTW_BACKWARD, FFTW_ESTIMATE);
	/////

	for(i_term=0;i_term<Nterms;i_term++){
		nu1 = -2.-alpha[i_term];
		nu2 = -2.-beta[i_term];
		for(i=0; i<N; i++){
			fb1[i] = f_unbias[i] / pow(x_full[i], nu1);
			fb2[i] = f_unbias[i] / pow(x_full[i], nu2);
		}
		g_m_vals(J1[i_term]+0.5, -0.5, eta_m, gl, halfN+1);

		fftw_execute(plan_forward);
		fftw_execute(plan_forward2);

		c_window(out, config->c_window_width, halfN);
		c_window(out2, config->c_window_width, halfN);

		// Do convolutions
		for(i=0; i<=halfN; i++) {
			out_pad1[i+halfN] = out[i] / (double)N * gl[i] ;
			// printf("gl:%lg, %lg\n", creal(gl[i]), cimag(gl[i]));
		}
		for(i=0; i<halfN; i++) {
			out_pad1[i] = conj(out_pad1[N-i]) ;
			// printf("gl:%e\n", gl[i]);
		}

		if(J1[i_term]!=J2[i_term]){
			g_m_vals(J2[i_term]+0.5, -0.5, eta_m, gl, halfN+1); // reuse gl array
		}

		for(i=0; i<=halfN; i++) {
			out_pad2[i+halfN] = out2[i] / (double)N * gl[i] ;
			// printf("gl:%e\n", gl[i]);
		}
		for(i=0; i<halfN; i++) {
			out_pad2[i] = conj(out_pad2[N-i]);
			// printf("gl:%e\n", gl[i]);
		}
		// fftconvolve(out_pad1, out_pad2, N+1, pads_convolve);
		fftconvolve_optimize(out_pad1, out_pad2, N+1, pads_convolve, a, b, a1, b1, c, pa, pb, pc);

		// convolution finished
		pads_convolve[N] = creal(pads_convolve[N]);

		for(i=0;i<=N;i++){
			h_part[i] = pads_convolve[i+N]; //C_h term in Eq.(2.21) in McEwen et al (2016)
											// but only take h = 0,1,2,...,N.
		}

		// p = -5.-2.*config->nu - alpha[i_term]-beta[i_term];
		g_m_vals(Jk[i_term]+0.5, -0.5, tau_l, fz, N+1);
		// f_z(0, tau_l, fz, N+1);

		for(i=0; i<=N; i++){
			out_vary[i] = h_part[i] * conj(fz[i]);
		}
		fftw_execute(plan_backward);

		for(i=0; i<N_original; i++){
			Fy[i_term][i] = out_ifft[2*(i+N_pad+N_extrap_low)] * pow(M_PI,1.5)/8. / x[i];
		}
	}


	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_forward2);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out2);
	fftw_free(out_vary);
	free(out_ifft);
	free(fb1);free(fb2);

	fftw_destroy_plan(pa);
	fftw_destroy_plan(pb);
	fftw_destroy_plan(pc);
	fftw_free(a);
	fftw_free(b);
	fftw_free(a1);
	fftw_free(b1);
	fftw_free(c);	
}




// For single (alpha,beta,ell) pair, not optimal, Deprecated!
void J_abl(double *x, double *fx, int alpha, int beta, long N, fastpt_config *config, int ell, double *Fy) {
	// x: k array, fx: Pin array

	long N_original = N;
	long N_pad = config->N_pad;
	long N_extrap_low = config->N_extrap_low;
	long N_extrap_high = config->N_extrap_high;
	N += (2*N_pad + N_extrap_low+N_extrap_high);

	if(N % 2) {printf("Please use even number of x !\n"); exit(0);}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i, j;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];
	
	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	double xi;
	int sign;
	if(N_extrap_low) {
		if(fx[0]==0) {
			printf("Can't log-extrapolate zero on the low side!\n");
			exit(1);
		}
		else if(fx[0]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[1]/fx[0]<=0) {printf("Log-extrapolation on the low side fails due to sign change!\n"); exit(1);}
		double dlnf_low = log(fx[1]/fx[0]);
		for(i=N_pad; i<N_pad+N_extrap_low; i++) {
			xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
			fb[i] = sign * exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
		}
	}
	for(i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) {
		fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
	}
	if(N_extrap_high) {
		if(fx[N_original-1]==0) {
			printf("Can't log-extrapolate zero on the high side!\n");
			exit(1);
		}
		else if(fx[N_original-1]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[N_original-1]/fx[N_original-2]<=0) {printf("Log-extrapolation on the high side fails due to sign change!\n"); exit(1);}
		double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
		for(i=N-N_pad-N_extrap_high; i<N-N_pad; i++) {
			xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
			fb[i] = sign * exp(log(fx[N_original-1]*sign) + (i- N_pad - N_extrap_low- N_original)*dlnf_high) / pow(xi, config->nu);
		}
	}

	fftw_complex *out, *out_vary;
	fftw_complex *out_pad1, *out_pad2;
	fftw_complex *pads_convolve;

	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	out_vary = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );

	out_pad1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	out_pad2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
	pads_convolve = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (2*N+1) );

	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window(out, config->c_window_width, halfN);
	// printf("out[1]:%.15e+i*(%.15e)\n", creal(out[1]), cimag(out[1]));

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * (2*N) );
	// double *out_ifft2;
	// out_ifft2 = malloc(sizeof(double) * N );

	plan_backward = fftw_plan_dft_c2r_1d(2*N, out_vary, out_ifft, FFTW_ESTIMATE);

	g_m_vals(ell+0.5, 1.5 + config->nu + alpha, eta_m, gl, halfN+1);

	// Do convolutions
	for(i=0; i<=halfN; i++) {
		out_pad1[i+halfN] = out[i] / (double)N * gl[i] ;
		// printf("gl:%e\n", gl[i]);
	}
	for(i=0; i<halfN; i++) {
		out_pad1[i] = conj(out_pad1[N-i]) ;
		// printf("gl:%e\n", gl[i]);
	}

	g_m_vals(ell+0.5, 1.5 + config->nu + beta, eta_m, gl, halfN+1);

	for(i=0; i<=halfN; i++) {
		out_pad2[i+halfN] = out[i] / (double)N * gl[i] ;
		// printf("gl:%e\n", gl[i]);
	}
	for(i=0; i<halfN; i++) {
		out_pad2[i] = conj(out_pad2[N-i]);
		// printf("gl:%e\n", gl[i]);
	}

	fftconvolve(out_pad1, out_pad2, N+1, pads_convolve); 

	// convolution finished
	pads_convolve[N] = creal(pads_convolve[N]);

	double complex h_part[N+1];
	double tau_l[N+1];
	for(i=0;i<=N;i++){
		h_part[i] = pads_convolve[i+N]; //C_h term in Eq.(2.21) in McEwen et al (2016)
										// but only take h = 0,1,2,...,N.
		tau_l[i] = 2.*M_PI / dlnx / N * i;
	}
	double complex fz[N+1];
	double p = -5.-2.*config->nu - alpha-beta;
	f_z(p+1, tau_l, fz, N+1);

	for(i=0; i<=N; i++){
		out_vary[i] = h_part[i] * conj(fz[i]) * cpow(2., I*tau_l[i]);
	}
	fftw_execute(plan_backward);

	sign = (ell%2? -1:1);
	for(i=0; i<N_original; i++){
		Fy[i] = out_ifft[2*(i+N_pad+N_extrap_low)] * sign / (M_PI*M_PI) * pow(2., 2.+2*config->nu+alpha+beta) * pow(x[i],-p-2.);
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	fftw_free(out_vary);
	free(out_ifft);
	free(fb);
}

void Pd1d2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[] = {0,0,1};
	int beta_ar[]  = {0,0,-1};
	int ell_ar[]   = {0,2,1};
	int isP13type_ar[] = {0,0,0};
	double coeff_A_ar[] = {2.*(17./21), 2.*(4./21), 2.};
	int Nterms = 3;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2d2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[] = {0};
	int beta_ar[]  = {0};
	int ell_ar[]   = {0};
	int isP13type_ar[] = {0};
	double coeff_A_ar[] = {2.};
	int Nterms = 1;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd1s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[] = {0,0,0,1,1};
	int beta_ar[]  = {0,0,0,-1,-1};
	int ell_ar[]   = {0,2,4,1,3};
	int isP13type_ar[] = {0,0,0,0,0};
	double coeff_A_ar[] = {2*(8./315.),2*(254./441.),2*(16./245.),2*(4./15.),2*(2./5.)};
	int Nterms = 5;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[] = {0};
	int beta_ar[]  = {0};
	int ell_ar[]   = {2};
	int isP13type_ar[] = {0};
	double coeff_A_ar[] = {2.*2./3.};
	int Nterms = 1;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Ps2s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[] = {0,0,0};
	int beta_ar[]  = {0,0,0};
	int ell_ar[]   = {0,2,4};
	int isP13type_ar[] = {0,0,0};
	double coeff_A_ar[] = {2.*(4./45.), 2*(8./63.), 2*(8./35.)};
	int Nterms = 3;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}


void IA_tt(double *k, double *Pin, long Nk, double *P_E, double *P_B){
	int Nterms_E=11, Nterms_B=10;

	int alpha_ar_E[] = {0,0,0,0,0,0,0,0,0,0,0};
	int beta_ar_E[]  = {0,0,0,0,0,0,0,0,0,0,0};
	int l1_ar_E[]    = {0,2,4,2,1,3,0,2,2,1,0};
	int l2_ar_E[]    = {0,0,0,2,1,1,0,0,2,1,0};
	int l_ar_E[]     = {0,0,0,0,1,1,2,2,2,3,4};
	double coeff_A_ar_E[] = {2.*(16./81.), 2*(713./1134.), 2*(38./315.), 2*(95./162), 2*(-107./60),\
							 2*(-19./15.), 2*(239./756.),  2*(11./9.),   2*(19./27.), 2*(-7./10.), \
							 2*(3./35)};

	int alpha_ar_B[] = {0,0,0,0,0,0,0,0,0,0};
	int beta_ar_B[]  = {0,0,0,0,0,0,0,0,0,0};
	int l1_ar_B[]    = {0,2,4,2,1,3,0,2,2,1};
	int l2_ar_B[]    = {0,0,0,2,1,1,0,0,2,1};
	int l_ar_B[]     = {0,0,0,0,1,1,2,2,2,3};
	double coeff_A_ar_B[] = {2.*(-41./405), 2*(-298./567), 2*(-32./315), 2*(-40./81), 2*(59./45),\
						 	 2*(16./15.),   2*(-2./9.),    2*(-20./27.), 2*(-16./27), 2*(2./5.)};

	int i;

	int Nterms_E_new =0, Nterms_B_new=0;
	for(i=0; i<Nterms_E; i++){
		Nterms_E_new += (l1_ar_E[i]+l2_ar_E[i]-abs(l1_ar_E[i]-l2_ar_E[i])+1) * (l1_ar_E[i]+l_ar_E[i]-abs(l1_ar_E[i]-l_ar_E[i])+1) * (l_ar_E[i]+l2_ar_E[i]-abs(l_ar_E[i]-l2_ar_E[i])+1);
	}
	for(i=0; i<Nterms_B; i++){
		Nterms_B_new += (l1_ar_B[i]+l2_ar_B[i]-abs(l1_ar_B[i]-l2_ar_B[i])+1) * (l1_ar_B[i]+l_ar_B[i]-abs(l1_ar_B[i]-l_ar_B[i])+1) * (l_ar_B[i]+l2_ar_B[i]-abs(l_ar_B[i]-l2_ar_B[i])+1);
	}

	int alpha_ar_E_new[Nterms_E_new], beta_ar_E_new[Nterms_E_new],J1_ar_E[Nterms_E_new], J2_ar_E[Nterms_E_new], Jk_ar_E[Nterms_E_new];
	int alpha_ar_B_new[Nterms_B_new], beta_ar_B_new[Nterms_B_new],J1_ar_B[Nterms_B_new], J2_ar_B[Nterms_B_new], Jk_ar_B[Nterms_B_new];
	double coeff_AB_ar_E[Nterms_E_new], coeff_AB_ar_B[Nterms_B_new];

	Nterms_E_new = J_table(alpha_ar_E, beta_ar_E, l1_ar_E, l2_ar_E, l_ar_E, coeff_A_ar_E, Nterms_E, alpha_ar_E_new, beta_ar_E_new, J1_ar_E, J2_ar_E, Jk_ar_E, coeff_AB_ar_E);
	Nterms_B_new = J_table(alpha_ar_B, beta_ar_B, l1_ar_B, l2_ar_B, l_ar_B, coeff_A_ar_B, Nterms_B, alpha_ar_B_new, beta_ar_B_new, J1_ar_B, J2_ar_B, Jk_ar_B, coeff_AB_ar_B);

	fastpt_tensor(alpha_ar_E_new, beta_ar_E_new, J1_ar_E, J2_ar_E, Jk_ar_E, coeff_AB_ar_E, Nterms_E_new, P_E, k, Pin, Nk);
	fastpt_tensor(alpha_ar_B_new, beta_ar_B_new, J1_ar_B, J2_ar_B, Jk_ar_B, coeff_AB_ar_B, Nterms_B_new, P_B, k, Pin, Nk);
}


void IA_ta(double *k, double *Pin, long Nk, double *P_dE1, double *P_dE2, double *P_0E0E, double *P_0B0B){
	int i;

	// deltaE1 term
	int Nterms_dE1=4;
	int alpha_ar_dE1[] = {0,0, 1, -1};
	int beta_ar_dE1[]  = {0,0,-1, 1};
	int l1_ar_dE1[]    = {0,0, 0,  0};
	int l2_ar_dE1[]    = {2,2,2,2};
	int l_ar_dE1[]     = {0,2,1,1};
	double coeff_A_ar_dE1[] = {2.*(17./21.), 2*(4./21.), 1., 1.};

	int Nterms_dE1_new =0;
	for(i=0; i<Nterms_dE1; i++){
		Nterms_dE1_new += (l1_ar_dE1[i]+l2_ar_dE1[i]-abs(l1_ar_dE1[i]-l2_ar_dE1[i])+1) * (l1_ar_dE1[i]+l_ar_dE1[i]-abs(l1_ar_dE1[i]-l_ar_dE1[i])+1) * (l_ar_dE1[i]+l2_ar_dE1[i]-abs(l_ar_dE1[i]-l2_ar_dE1[i])+1);
	}

	int alpha_ar_dE1_new[Nterms_dE1_new], beta_ar_dE1_new[Nterms_dE1_new],J1_ar_dE1[Nterms_dE1_new], J2_ar_dE1[Nterms_dE1_new], Jk_ar_dE1[Nterms_dE1_new];
	double coeff_AB_ar_dE1[Nterms_dE1_new];

	Nterms_dE1_new = J_table(alpha_ar_dE1, beta_ar_dE1, l1_ar_dE1, l2_ar_dE1, l_ar_dE1, coeff_A_ar_dE1, Nterms_dE1, alpha_ar_dE1_new, beta_ar_dE1_new, J1_ar_dE1, J2_ar_dE1, Jk_ar_dE1, coeff_AB_ar_dE1);
	fastpt_tensor(alpha_ar_dE1_new, beta_ar_dE1_new, J1_ar_dE1, J2_ar_dE1, Jk_ar_dE1, coeff_AB_ar_dE1, Nterms_dE1_new, P_dE1, k, Pin, Nk);

	// 0E0E term
	int Nterms_0E0E=4;
	int alpha_ar_0E0E[] = {0,0,0,0};
	int beta_ar_0E0E[]  = {0,0,0,0};
	int l1_ar_0E0E[]    = {0,2,2,0};
	int l2_ar_0E0E[]    = {0,0,2,4};
	int l_ar_0E0E[]     = {0,0,0,0};
	double coeff_A_ar_0E0E[] = {29./90., 5./63., 19./18., 19./35};

	int Nterms_0E0E_new=0;
	for(i=0; i<Nterms_0E0E; i++){
		Nterms_0E0E_new += (l1_ar_0E0E[i]+l2_ar_0E0E[i]-abs(l1_ar_0E0E[i]-l2_ar_0E0E[i])+1) * (l1_ar_0E0E[i]+l_ar_0E0E[i]-abs(l1_ar_0E0E[i]-l_ar_0E0E[i])+1) * (l_ar_0E0E[i]+l2_ar_0E0E[i]-abs(l_ar_0E0E[i]-l2_ar_0E0E[i])+1);
	}
	int alpha_ar_0E0E_new[Nterms_0E0E_new], beta_ar_0E0E_new[Nterms_0E0E_new],J1_ar_0E0E[Nterms_0E0E_new], J2_ar_0E0E[Nterms_0E0E_new], Jk_ar_0E0E[Nterms_0E0E_new];
	double coeff_AB_ar_0E0E[Nterms_0E0E_new];
	Nterms_0E0E_new = J_table(alpha_ar_0E0E, beta_ar_0E0E, l1_ar_0E0E, l2_ar_0E0E, l_ar_0E0E, coeff_A_ar_0E0E, Nterms_0E0E, alpha_ar_0E0E_new, beta_ar_0E0E_new, J1_ar_0E0E, J2_ar_0E0E, Jk_ar_0E0E, coeff_AB_ar_0E0E);
	fastpt_tensor(alpha_ar_0E0E_new, beta_ar_0E0E_new, J1_ar_0E0E, J2_ar_0E0E, Jk_ar_0E0E, coeff_AB_ar_0E0E, Nterms_0E0E_new, P_0E0E, k, Pin, Nk);

	// 0B0B term
	int Nterms_0B0B=5;
	int alpha_ar_0B0B[] = {0,0,0,0,0};
	int beta_ar_0B0B[]  = {0,0,0,0,0};
	int l1_ar_0B0B[]    = {0,2,2,0,1};
	int l2_ar_0B0B[]    = {0,0,2,4,1};
	int l_ar_0B0B[]     = {0,0,0,0,1};
	double coeff_A_ar_0B0B[] = {2./45, -44./63, -8./9, -16./35, 2.};

	int Nterms_0B0B_new=0;
	for(i=0; i<Nterms_0B0B; i++){
		Nterms_0B0B_new += (l1_ar_0B0B[i]+l2_ar_0B0B[i]-abs(l1_ar_0B0B[i]-l2_ar_0B0B[i])+1) * (l1_ar_0B0B[i]+l_ar_0B0B[i]-abs(l1_ar_0B0B[i]-l_ar_0B0B[i])+1) * (l_ar_0B0B[i]+l2_ar_0B0B[i]-abs(l_ar_0B0B[i]-l2_ar_0B0B[i])+1);
	}
	int alpha_ar_0B0B_new[Nterms_0B0B_new], beta_ar_0B0B_new[Nterms_0B0B_new],J1_ar_0B0B[Nterms_0B0B_new], J2_ar_0B0B[Nterms_0B0B_new], Jk_ar_0B0B[Nterms_0B0B_new];
	double coeff_AB_ar_0B0B[Nterms_0B0B_new];
	Nterms_0B0B_new = J_table(alpha_ar_0B0B, beta_ar_0B0B, l1_ar_0B0B, l2_ar_0B0B, l_ar_0B0B, coeff_A_ar_0B0B, Nterms_0B0B, alpha_ar_0B0B_new, beta_ar_0B0B_new, J1_ar_0B0B, J2_ar_0B0B, Jk_ar_0B0B, coeff_AB_ar_0B0B);
	fastpt_tensor(alpha_ar_0B0B_new, beta_ar_0B0B_new, J1_ar_0B0B, J2_ar_0B0B, Jk_ar_0B0B, coeff_AB_ar_0B0B, Nterms_0B0B_new, P_0B0B, k, Pin, Nk);

	// deltaE2 term
	double exps[2*Nk-1], f[2*Nk-1];
	double dL = log(k[1]/k[0]);
	long Ncut = floor(3./dL);
	double r;
	for(i=0; i<2*Nk-1; i++){
		exps[i] = exp(-dL*(i-Nk+1));
	}

	for(i=0; i<Nk-1-Ncut; i++){
		r = exps[i];
		f[i] = r* ( 768./7 - 256/(7293.*pow(r,10)) - 256/(3003.*pow(r,8)) - 256/(1001.*pow(r,6)) - 256/(231.*pow(r,4)) - 256/(21.*r*r)  );
	}
	for( ; i<Nk-1; i++){
		r = exps[i];
		f[i] = r* ( 30. + 146*r*r - 110*pow(r,4) + 30*pow(r,6) + log(fabs(r-1.)/(r+1.))*(15./r - 60.*r + 90*pow(r,3) - 60*pow(r,5) + 15*pow(r,7))  );
	}
	for(i=Nk; i<Nk-1+Ncut; i++){
		r = exps[i];
		f[i] = r* ( 30. + 146*r*r - 110*pow(r,4) + 30*pow(r,6) + log(fabs(r-1.)/(r+1.))*(15./r - 60.*r + 90*pow(r,3) - 60*pow(r,5) + 15*pow(r,7))  );
	}
	for( ; i<2*Nk-1; i++){
		r = exps[i];
		f[i] = r* ( 256*r*r - 256*pow(r,4) + (768*pow(r,6))/7. - (256*pow(r,8))/21. - (256*pow(r,10))/231. - (256*pow(r,12))/1001. - (256*pow(r,14))/3003.  );
	}
	f[Nk-1] = 96.;
	double g[3*Nk-2];
	fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
	for(i=0; i<Nk; i++){
		P_dE2[i] = 2.* pow(k[i],3)/(896.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL; 
	}
}

void IA_mix(double *k, double *Pin, long Nk, double *P_A, double *P_B, double *P_DEE, double *P_DBB){
	int i;

	// A term
	int Nterms_A=13;
	int alpha_ar_A[] = {0,0,0,0,0,0,0, 1,1,1,1,1,1};
	int beta_ar_A[]  = {0,0,0,0,0,0,0, -1,-1,-1,-1,-1,-1};
	int l1_ar_A[]    = {0,2,0,2,1,1,0, 0,2,1,1,0,0};
	int l2_ar_A[]    = {0,0,0,0,1,1,0, 0,0,1,1,2,0};
	int l_ar_A[]     = {0,0,2,2,1,3,4, 1,1,0,2,1,3};
	double coeff_A_ar_A[] = {2.*(-31./210.), 2*(-34./63), 2*(-47./147), 2*(-8./63),2*(93./70), 2*(6./35), 2*(-8./245),\
							2.*(-3./10),2.*(-1./3),2.*(1./2),2.*(1.),2.*(-1./3),2.*(-1./5)};

	int Nterms_A_new =0;
	for(i=0; i<Nterms_A; i++){
		Nterms_A_new += (l1_ar_A[i]+l2_ar_A[i]-abs(l1_ar_A[i]-l2_ar_A[i])+1) * (l1_ar_A[i]+l_ar_A[i]-abs(l1_ar_A[i]-l_ar_A[i])+1) * (l_ar_A[i]+l2_ar_A[i]-abs(l_ar_A[i]-l2_ar_A[i])+1);
	}

	int alpha_ar_A_new[Nterms_A_new], beta_ar_A_new[Nterms_A_new],J1_ar_A[Nterms_A_new], J2_ar_A[Nterms_A_new], Jk_ar_A[Nterms_A_new];
	double coeff_AB_ar_A[Nterms_A_new];

	Nterms_A_new = J_table(alpha_ar_A, beta_ar_A, l1_ar_A, l2_ar_A, l_ar_A, coeff_A_ar_A, Nterms_A, alpha_ar_A_new, beta_ar_A_new, J1_ar_A, J2_ar_A, Jk_ar_A, coeff_AB_ar_A);
	fastpt_tensor(alpha_ar_A_new, beta_ar_A_new, J1_ar_A, J2_ar_A, Jk_ar_A, coeff_AB_ar_A, Nterms_A_new, P_A, k, Pin, Nk);

	// D_EE term
	int Nterms_DEE=8;
	int alpha_ar_DEE[] = {0,0,0,0, 0,0,0,0};
	int beta_ar_DEE[]  = {0,0,0,0, 0,0,0,0};
	int l1_ar_DEE[]    = {0,2,4,0, 2,1,3,2};
	int l2_ar_DEE[]    = {0,0,0,0, 0,1,1,2};
	int l_ar_DEE[]     = {0,0,0,2, 2,1,1,0};
	double coeff_A_ar_DEE[] = {2.*(-43./540), 2*(-167./756), 2*(-19./105), 2*(1./18),\
							2*(-7./18), 2*(11./20), 2*(19./20), 2.*(-19./54)};

	int Nterms_DEE_new=0;
	for(i=0; i<Nterms_DEE; i++){
		Nterms_DEE_new += (l1_ar_DEE[i]+l2_ar_DEE[i]-abs(l1_ar_DEE[i]-l2_ar_DEE[i])+1) * (l1_ar_DEE[i]+l_ar_DEE[i]-abs(l1_ar_DEE[i]-l_ar_DEE[i])+1) * (l_ar_DEE[i]+l2_ar_DEE[i]-abs(l_ar_DEE[i]-l2_ar_DEE[i])+1);
	}
	int alpha_ar_DEE_new[Nterms_DEE_new], beta_ar_DEE_new[Nterms_DEE_new],J1_ar_DEE[Nterms_DEE_new], J2_ar_DEE[Nterms_DEE_new], Jk_ar_DEE[Nterms_DEE_new];
	double coeff_AB_ar_DEE[Nterms_DEE_new];
	Nterms_DEE_new = J_table(alpha_ar_DEE, beta_ar_DEE, l1_ar_DEE, l2_ar_DEE, l_ar_DEE, coeff_A_ar_DEE, Nterms_DEE, alpha_ar_DEE_new, beta_ar_DEE_new, J1_ar_DEE, J2_ar_DEE, Jk_ar_DEE, coeff_AB_ar_DEE);
	fastpt_tensor(alpha_ar_DEE_new, beta_ar_DEE_new, J1_ar_DEE, J2_ar_DEE, Jk_ar_DEE, coeff_AB_ar_DEE, Nterms_DEE_new, P_DEE, k, Pin, Nk);

	// D_BB term
	int Nterms_DBB=8;
	int alpha_ar_DBB[] = {0,0,0,0, 0,0,0,0};
	int beta_ar_DBB[]  = {0,0,0,0, 0,0,0,0};
	int l1_ar_DBB[]    = {0,2,4,0, 2,1,3,2};
	int l2_ar_DBB[]    = {0,0,0,0, 0,1,1,2};
	int l_ar_DBB[]     = {0,0,0,2, 2,1,1,0};
	double coeff_A_ar_DBB[] = {2.*(13./135), 2*(86./189), 2*(16./105), 2*(2./9),\
							2*(4./9), 2*(-13./15), 2*(-4./5), 2.*(8./27)};

	int Nterms_DBB_new=0;
	for(i=0; i<Nterms_DBB; i++){
		Nterms_DBB_new += (l1_ar_DBB[i]+l2_ar_DBB[i]-abs(l1_ar_DBB[i]-l2_ar_DBB[i])+1) * (l1_ar_DBB[i]+l_ar_DBB[i]-abs(l1_ar_DBB[i]-l_ar_DBB[i])+1) * (l_ar_DBB[i]+l2_ar_DBB[i]-abs(l_ar_DBB[i]-l2_ar_DBB[i])+1);
	}
	int alpha_ar_DBB_new[Nterms_DBB_new], beta_ar_DBB_new[Nterms_DBB_new],J1_ar_DBB[Nterms_DBB_new], J2_ar_DBB[Nterms_DBB_new], Jk_ar_DBB[Nterms_DBB_new];
	double coeff_AB_ar_DBB[Nterms_DBB_new];
	Nterms_DBB_new = J_table(alpha_ar_DBB, beta_ar_DBB, l1_ar_DBB, l2_ar_DBB, l_ar_DBB, coeff_A_ar_DBB, Nterms_DBB, alpha_ar_DBB_new, beta_ar_DBB_new, J1_ar_DBB, J2_ar_DBB, Jk_ar_DBB, coeff_AB_ar_DBB);
	fastpt_tensor(alpha_ar_DBB_new, beta_ar_DBB_new, J1_ar_DBB, J2_ar_DBB, Jk_ar_DBB, coeff_AB_ar_DBB, Nterms_DBB_new, P_DBB, k, Pin, Nk);

	// B term
	double exps[2*Nk-1], f[2*Nk-1];
	double dL = log(k[1]/k[0]);
	long Ncut = floor(3./dL);
	double r;
	for(i=0; i<2*Nk-1; i++){
		exps[i] = exp(-dL*(i-Nk+1));
	}

	for(i=0; i<Nk-1-Ncut; i++){
		r = exps[i];
		f[i] = r* (-16./147 - 16/(415701.*pow(r,12)) - 32/(357357.*pow(r,10)) - 16/(63063.*pow(r,8)) - 64/(63063.*pow(r,6)) - 16/(1617.*pow(r,4)) + 32/(441.*r*r) )/2.;
	}
	for( ; i<Nk-1; i++){
		r = exps[i];
		f[i] = r* ((2.* r * (225.- 600.* r*r + 1198.* pow(r,4) - 600.* pow(r,6) + 225.* pow(r,8)) + \
    				225.* pow((r*r - 1.),4) * (r*r + 1.) * log(fabs(r-1)/(r+1)) )/(20160.* pow(r,3)) - 29./315*r*r )/2.;
	}
	for(i=Nk; i<Nk-1+Ncut; i++){
		r = exps[i];
		f[i] = r* ((2.* r * (225.- 600.* r*r + 1198.* pow(r,4) - 600.* pow(r,6) + 225.* pow(r,8)) + \
    				225.* pow((r*r - 1.),4) * (r*r + 1.) * log(fabs(r-1)/(r+1)) )/(20160.* pow(r,3)) - 29./315*r*r )/2.;
	}
	for( ; i<2*Nk-1; i++){
		r = exps[i];
		f[i] = r* ( (-16*pow(r,4))/147. + (32*pow(r,6))/441. - (16*pow(r,8))/1617. - (64*pow(r,10))/63063. - 16*pow(r,12)/63063. - (32*pow(r,14))/357357. - (16*pow(r,16))/415701. )/2.;
	}
	f[Nk-1] = -1./42.;
	double g[3*Nk-2];
	fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
	for(i=0; i<Nk; i++){
		P_B[i] = 4.* pow(k[i],3)/(2.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL; 
	}
}
