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
	config.nu = -2.; config.c_window_width = 0.25; config.N_pad = 1000;
	config.N_extrap_low = 00; config.N_extrap_high = 0;

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
	config.c_window_width = 0.25; config.N_pad = 1000;
	config.N_extrap_low = 00; config.N_extrap_high = 0;

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


int main(int argc, char const *argv[])
{
	FILE *finput;
	finput = fopen("Pk_test.dat", "r");
	long Nk = 3000;
	long line_num;
	double k[Nk], Pin[Nk];
	double dummy; 
	if(finput == NULL) 
	{printf("File not found\n");}
	else
	{
		line_num = 0;
		while(!feof(finput)) {
			fscanf(finput, "%lg %lg %lg %lg", &(k[line_num]), &(Pin[line_num]), &dummy, &dummy );
			line_num++;
		}
		fclose(finput);
	}

	// int alpha_ar[] = {0,0,0,2,1,1,2};
	// int beta_ar[]  = {0,0,0,-2,-1,-1,-2};
	// int ell_ar[]   = {0,2,4,2,1,3,0};


	double Pout[Nk];
	double P_E[Nk],P_B[Nk];

	clock_t t1, t2;
	t1 = clock();
	// Pd1d2(k, Pin, Nk, Pout);
	// Pd2d2(k, Pin, Nk, Pout);
	// Pd1s2(k, Pin, Nk, Pout);
	// Pd2s2(k, Pin, Nk, Pout);
	// Ps2s2(k, Pin, Nk, Pout);

	IA_tt(k, Pin, Nk, P_E, P_B);
	t2 = clock();
	printf("time: %lg\n", (double)(t2 - t1) / CLOCKS_PER_SEC);



	FILE *fout;
	fout = fopen("out.txt", "w");
	for(line_num=0; line_num<Nk; line_num++){
		// fprintf(fout, "%lg %lg %lg\n", k[line_num], Pin[line_num], Pout[line_num]);
		fprintf(fout, "%lg %lg %lg %lg\n", k[line_num], Pin[line_num], P_E[line_num], P_B[line_num]);
	}
	fclose(fout);
	return 0;
}