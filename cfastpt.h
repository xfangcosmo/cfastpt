typedef struct fastpt_config {
	double nu;
	double c_window_width;
	long N_pad;
	long N_extrap_low;
	long N_extrap_high;
} fastpt_config;

void fastpt_scalar(int *alpha_ar, int *beta_ar, int *ell_ar, int *isP13type_ar, double *coeff_A_ar, int Nterms, double *Pout, double *k, double *Pin, int Nk);

void J_abl_ar(double *x, double *fx, long N, int *alpha, int *beta, int *ell, int *isP13type, int Nterms, fastpt_config *config, double **Fy);

void J_abl(double *x, double *fx, int alpha, int beta, long N, fastpt_config *config, int ell, double *Fy);

typedef struct fastpt_todo {
	int isScalar;
	double *alpha;
	double *beta;
	double *ell;
	int *isP13type;
	double *coeff_ar;
	int Nterms;
} fastpt_todo;

typedef struct fastpt_todolist {
	fastpt_todo *fastpt_todo;
	int N_todo;
} fastpt_todolist;


void Pd1d2(double *k, double *Pin, long Nk, double *Pout);
void Pd2d2(double *k, double *Pin, long Nk, double *Pout);
void Pd1s2(double *k, double *Pin, long Nk, double *Pout);
void Pd2s2(double *k, double *Pin, long Nk, double *Pout);
void Ps2s2(double *k, double *Pin, long Nk, double *Pout);