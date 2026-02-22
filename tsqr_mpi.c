#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Fortran LAPACK/BLAS symbols */
extern void dgeqrf_(const int* m, const int* n, double* a, const int* lda,
    double* tau, double* work, const int* lwork, int* info);

extern void dorgqr_(const int* m, const int* n, const int* k,
    double* a, const int* lda, const double* tau,
    double* work, const int* lwork, int* info);

extern void dgemm_(const char* TRANSA, const char* TRANSB,
    const int* M, const int* N, const int* K,
    const double* ALPHA, const double* A, const int* LDA,
    const double* B, const int* LDB,
    const double* BETA, double* C, const int* LDC);

/* column-major indexing */
static inline int idx(int i, int j, int ld) { return i + j * ld; }

static void die_if(int cond, const char* msg, MPI_Comm comm) {
    if (cond) {
        int r;
        MPI_Comm_rank(comm, &r);
        if (r == 0) fprintf(stderr, "ERROR: %s\n", msg);
        MPI_Abort(comm, 1);
    }
}

/* Frobenius norm for column-major matrix A (m x n, leading dim lda) */
static double frob_norm(const double* A, int m, int n, int lda) {
    double s = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double v = A[idx(i, j, lda)];
            s += v * v;
        }
    }
    return sqrt(s);
}

/*  Deterministic matrix generation (reproducible across ranks) */

static inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double rand_entry(unsigned long long seed, long long i, long long j) {
    unsigned long long x = seed;
    x ^= (unsigned long long)i * 0xD1342543DE82EF95ULL;
    x ^= (unsigned long long)j * 0xA24BAED4963EE407ULL;
    x = splitmix64(x);
    /* to [0,1) from top bits */
    double u = (double)((x >> 11) & 0x1fffffffffffffULL) / (double)(0x20000000000000ULL);
    return 2.0 * u - 1.0; /* [-1,1] */
}

static void fill_A_local(double* A_local, int mloc, int n, long long global_row0,
    unsigned long long seed) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < mloc; ++i) {
            A_local[idx(i, j, mloc)] = rand_entry(seed, global_row0 + i, j);
        }
    }
}

static void fill_A_full(double* A_full, int m, int n, unsigned long long seed) {
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            A_full[idx(i, j, m)] = rand_entry(seed, i, j);
        }
    }
}

/* Linear algebra helpers */

static void matmul_inplace_right(double* Q, int m, int n,
    const double* S /* n x n */, int lds) {
    /* Q := Q * S */
    double* C = (double*)malloc((size_t)m * (size_t)n * sizeof(double));
    if (!C) { fprintf(stderr, "alloc failed\n"); exit(1); }

    const char Nch = 'N';
    const double one = 1.0, zero = 0.0;
    dgemm_(&Nch, &Nch, &m, &n, &n, &one, Q, &m, S, &lds, &zero, C, &m);

    memcpy(Q, C, (size_t)m * (size_t)n * sizeof(double));
    free(C);
}

static void local_qr_thin(double* A /* m x n */, int m, int n,
    double* Q_out /* m x n */,
    double* R_out /* n x n */) {
    memcpy(Q_out, A, (size_t)m * (size_t)n * sizeof(double));

    int info = 0;
    int lda = m;
    int k = n;
    double* tau = (double*)malloc((size_t)n * sizeof(double));
    if (!tau) { fprintf(stderr, "alloc failed\n"); exit(1); }

    /* workspace query for dgeqrf */
    int lwork = -1;
    double wkopt = 0.0;
    dgeqrf_(&m, &n, Q_out, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dgeqrf query failed info=%d\n", info); exit(1); }
    lwork = (int)wkopt;
    double* work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) { fprintf(stderr, "alloc failed\n"); exit(1); }

    dgeqrf_(&m, &n, Q_out, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dgeqrf failed info=%d\n", info); exit(1); }

    /* extract R */
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            R_out[idx(i, j, n)] = (i <= j) ? Q_out[idx(i, j, m)] : 0.0;
        }
    }

    /* workspace query for dorgqr */
    free(work);
    lwork = -1;
    dorgqr_(&m, &n, &k, Q_out, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dorgqr query failed info=%d\n", info); exit(1); }
    lwork = (int)wkopt;
    work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) { fprintf(stderr, "alloc failed\n"); exit(1); }

    dorgqr_(&m, &n, &k, Q_out, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dorgqr failed info=%d\n", info); exit(1); }

    free(work);
    free(tau);
}

static void qr_of_stacked_R(const double* R_top /* n x n */,
    const double* R_bot /* n x n */,
    int n,
    double* Q_stack /* (2n) x n */,
    double* R_out   /* n x n */) {
    int m = 2 * n, lda = 2 * n;
    double* A = Q_stack;

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) A[idx(i, j, lda)] = R_top[idx(i, j, n)];
        for (int i = 0; i < n; ++i) A[idx(i + n, j, lda)] = R_bot[idx(i, j, n)];
    }

    int info = 0;
    double* tau = (double*)malloc((size_t)n * sizeof(double));
    if (!tau) { fprintf(stderr, "alloc failed\n"); exit(1); }

    int lwork = -1;
    double wkopt = 0.0;
    dgeqrf_(&m, &n, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dgeqrf2 query failed info=%d\n", info); exit(1); }
    lwork = (int)wkopt;
    double* work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) { fprintf(stderr, "alloc failed\n"); exit(1); }

    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dgeqrf2 failed info=%d\n", info); exit(1); }

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            R_out[idx(i, j, n)] = (i <= j) ? A[idx(i, j, lda)] : 0.0;
        }
    }

    free(work);
    int k = n;
    lwork = -1;
    dorgqr_(&m, &n, &k, A, &lda, tau, &wkopt, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dorgqr2 query failed info=%d\n", info); exit(1); }
    lwork = (int)wkopt;
    work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) { fprintf(stderr, "alloc failed\n"); exit(1); }

    dorgqr_(&m, &n, &k, A, &lda, tau, work, &lwork, &info);
    if (info != 0) { fprintf(stderr, "dorgqr2 failed info=%d\n", info); exit(1); }

    free(work);
    free(tau);
}

/* TSQR with timing breakdown  */
typedef struct {
    double t_localqr;
    double t_reduce; 
    double t_apply;
    double t_total;
} tsqr_times_t;

/* TSQR specialized for exactly 4 ranks; returns R on rank 0 and overwrites Q_local */
static void TSQR_4(double* A_local, int mloc, int n, MPI_Comm comm,
    double* Q_local, double* R_final /* rank0 only */,
    tsqr_times_t* out_rank0 /* valid on rank0 */) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    die_if(size != 4, "This TSQR_4 implementation requires exactly 4 MPI ranks.", comm);

    MPI_Barrier(comm);
    double t_all0 = MPI_Wtime();

    /* Stage 1: local QR */
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    double* R_leaf = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    die_if(!R_leaf, "alloc failed", comm);
    local_qr_thin(A_local, mloc, n, Q_local, R_leaf);

    double t1 = MPI_Wtime();
    double t_local = t1 - t0;

    /* Stage 2 + 3 reductions */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();

    double* Q01 = NULL; /* (2n x n) on rank0 */
    double* Q11 = NULL; /* (2n x n) on rank2 */
    double* R01 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    double* R11 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    die_if(!R01 || !R11, "alloc failed", comm);

    if (rank == 1) MPI_Send(R_leaf, n * n, MPI_DOUBLE, 0, 100, comm);
    if (rank == 3) MPI_Send(R_leaf, n * n, MPI_DOUBLE, 2, 200, comm);

    if (rank == 0) {
        double* R1 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        Q01 = (double*)malloc((size_t)(2 * n) * (size_t)n * sizeof(double));
        die_if(!R1 || !Q01, "alloc failed", comm);
        MPI_Recv(R1, n * n, MPI_DOUBLE, 1, 100, comm, MPI_STATUS_IGNORE);
        qr_of_stacked_R(R_leaf, R1, n, Q01, R01);
        free(R1);
    }

    if (rank == 2) {
        double* R3 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        Q11 = (double*)malloc((size_t)(2 * n) * (size_t)n * sizeof(double));
        die_if(!R3 || !Q11, "alloc failed", comm);
        MPI_Recv(R3, n * n, MPI_DOUBLE, 3, 200, comm, MPI_STATUS_IGNORE);
        qr_of_stacked_R(R_leaf, R3, n, Q11, R11);
        free(R3);
        MPI_Send(R11, n * n, MPI_DOUBLE, 0, 300, comm);
    }

    if (rank == 0) {
        MPI_Recv(R11, n * n, MPI_DOUBLE, 2, 300, comm, MPI_STATUS_IGNORE);
    }

    double* Q02 = NULL; /* (2n x n) on rank0 */
    if (rank == 0) {
        Q02 = (double*)malloc((size_t)(2 * n) * (size_t)n * sizeof(double));
        die_if(!Q02, "alloc failed", comm);
        qr_of_stacked_R(R01, R11, n, Q02, R_final);
    }

    t1 = MPI_Wtime();
    double t_reduce = t1 - t0;

    /* Apply small factors */
    MPI_Barrier(comm);
    t0 = MPI_Wtime();

    double* S2 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    double* S3 = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    die_if(!S2 || !S3, "alloc failed", comm);

    if (rank == 0) {
        /* Extract Q02_top/bot and Q01_top/bot */
        double* Q02_top = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        double* Q02_bot = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        double* Q01_top = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        double* Q01_bot = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        die_if(!Q02_top || !Q02_bot || !Q01_top || !Q01_bot, "alloc failed", comm);

        int ld = 2 * n;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Q02_top[idx(i, j, n)] = Q02[idx(i, j, ld)];
                Q02_bot[idx(i, j, n)] = Q02[idx(i + n, j, ld)];
                Q01_top[idx(i, j, n)] = Q01[idx(i, j, ld)];
                Q01_bot[idx(i, j, n)] = Q01[idx(i + n, j, ld)];
            }
        }

        memcpy(S2, Q01_top, (size_t)n * (size_t)n * sizeof(double));
        memcpy(S3, Q02_top, (size_t)n * (size_t)n * sizeof(double));

        MPI_Send(Q01_bot, n * n, MPI_DOUBLE, 1, 401, comm);
        MPI_Send(Q02_top, n * n, MPI_DOUBLE, 1, 402, comm);
        MPI_Send(Q02_bot, n * n, MPI_DOUBLE, 2, 403, comm);

        free(Q02_top); free(Q02_bot);
        free(Q01_top); free(Q01_bot);
    }

    if (rank == 1) {
        MPI_Recv(S2, n * n, MPI_DOUBLE, 0, 401, comm, MPI_STATUS_IGNORE);
        MPI_Recv(S3, n * n, MPI_DOUBLE, 0, 402, comm, MPI_STATUS_IGNORE);
    }

    if (rank == 2) {
        MPI_Recv(S3, n * n, MPI_DOUBLE, 0, 403, comm, MPI_STATUS_IGNORE);

        /* Extract Q11_top/bot; send bot to rank3 */
        double* Q11_top = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        double* Q11_bot = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        die_if(!Q11_top || !Q11_bot, "alloc failed", comm);

        int ld = 2 * n;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Q11_top[idx(i, j, n)] = Q11[idx(i, j, ld)];
                Q11_bot[idx(i, j, n)] = Q11[idx(i + n, j, ld)];
            }
        }
        memcpy(S2, Q11_top, (size_t)n * (size_t)n * sizeof(double));

        MPI_Send(Q11_bot, n * n, MPI_DOUBLE, 3, 404, comm);
        MPI_Send(S3, n * n, MPI_DOUBLE, 3, 405, comm);

        free(Q11_top);
        free(Q11_bot);
    }

    if (rank == 3) {
        MPI_Recv(S2, n * n, MPI_DOUBLE, 2, 404, comm, MPI_STATUS_IGNORE);
        MPI_Recv(S3, n * n, MPI_DOUBLE, 2, 405, comm, MPI_STATUS_IGNORE);
    }

    matmul_inplace_right(Q_local, mloc, n, S2, n);
    matmul_inplace_right(Q_local, mloc, n, S3, n);

    t1 = MPI_Wtime();
    double t_apply = t1 - t0;

    MPI_Barrier(comm);
    double t_total = MPI_Wtime() - t_all0;

    /* Reduce by MAX across ranks */
    double local[4] = { t_local, t_reduce, t_apply, t_total };
    double maxv[4] = { 0,0,0,0 };
    MPI_Reduce(local, maxv, 4, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0 && out_rank0) {
        out_rank0->t_localqr = maxv[0];
        out_rank0->t_reduce = maxv[1];
        out_rank0->t_apply = maxv[2];
        out_rank0->t_total = maxv[3];
    }

    free(S2); free(S3);
    free(R_leaf);
    free(R01); free(R11);
    if (rank == 0) { free(Q01); free(Q02); }
    if (rank == 2) { free(Q11); }
}

/* Correctness: compute rel_res and orth AFTER timing */
static int compute_relres_orth(int m, int n, int mloc,
    double* Q_local, double* R_final,
    MPI_Comm comm, unsigned long long seed,
    double* rel_res_out, double* orth_out) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    /* gather Q to rank0 */
    double* Q_full = NULL;
    if (rank == 0) {
        Q_full = (double*)malloc((size_t)m * (size_t)n * sizeof(double));
        if (!Q_full) return 0;
    }
    for (int j = 0; j < n; ++j) {
        MPI_Gather(&Q_local[idx(0, j, mloc)], mloc, MPI_DOUBLE,
            rank == 0 ? &Q_full[idx(0, j, m)] : NULL, mloc, MPI_DOUBLE,
            0, comm);
    }

    if (rank == 0) {
        /* build A_full deterministically on rank0 */
        double* A_full = (double*)malloc((size_t)m * (size_t)n * sizeof(double));
        double* Aapprox = (double*)malloc((size_t)m * (size_t)n * sizeof(double));
        double* QtQ = (double*)calloc((size_t)n * (size_t)n, sizeof(double));
        if (!A_full || !Aapprox || !QtQ) {
            free(A_full); free(Aapprox); free(QtQ); free(Q_full);
            return 0;
        }
        fill_A_full(A_full, m, n, seed);

        const char Nch = 'N';
        const char Tch = 'T';
        const double one = 1.0, zero = 0.0;

        /* Aapprox = Q*R */
        dgemm_(&Nch, &Nch, &m, &n, &n, &one, Q_full, &m, R_final, &n, &zero, Aapprox, &m);

        /* residual = A - Aapprox (store in Aapprox) */
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                Aapprox[idx(i, j, m)] = A_full[idx(i, j, m)] - Aapprox[idx(i, j, m)];
            }
        }

        double nA = frob_norm(A_full, m, n, m);
        double nRes = frob_norm(Aapprox, m, n, m);
        *rel_res_out = nRes / (nA > 0.0 ? nA : 1.0);

        /* QtQ = Q^T Q */
        dgemm_(&Tch, &Nch, &n, &n, &m, &one, Q_full, &m, Q_full, &m, &zero, QtQ, &n);
        for (int i = 0; i < n; ++i) QtQ[idx(i, i, n)] -= 1.0;
        *orth_out = frob_norm(QtQ, n, n, n);

        free(A_full);
        free(Aapprox);
        free(QtQ);
        free(Q_full);
    }

    /* broadcast to all ranks (so everyone has the numbers; rank0 prints) */
    MPI_Bcast(rel_res_out, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(orth_out, 1, MPI_DOUBLE, 0, comm);
    return 1;
}

/* stats */
static double mean(const double* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s / (double)n;
}

static double stddev(const double* x, int n) {
    if (n <= 1) return 0.0;
    double mu = mean(x, n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - mu;
        s += d * d;
    }
    return sqrt(s / (double)(n - 1));
}

/* run one (m,n) point, print one CSV row */
static void run_point(const char* mode, int m, int n, int reps, int warmup,
    MPI_Comm comm, unsigned long long seed) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    if (m <= 0 || n <= 0 || m < n || (m % 4 != 0)) {
        if (rank == 0) {
            printf("%s,%d,%d,,%d,%d,,,,,,,SKIP_BAD\n", mode, m, n, reps, warmup);
        }
        return;
    }

    int mloc = m / 4;
    long long global_row0 = (long long)rank * (long long)mloc;

    double* A_local = (double*)malloc((size_t)mloc * (size_t)n * sizeof(double));
    double* Q_local = (double*)malloc((size_t)mloc * (size_t)n * sizeof(double));
    die_if(!A_local || !Q_local, "alloc failed", comm);

    fill_A_local(A_local, mloc, n, global_row0, seed);

    double* R_final = NULL;
    if (rank == 0) {
        R_final = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
        die_if(!R_final, "alloc failed", comm);
    }

    /* warmup  */
    for (int w = 0; w < warmup; ++w) {
        tsqr_times_t tmp;
        TSQR_4(A_local, mloc, n, comm, Q_local, R_final, &tmp);
    }

    double* ttot = (double*)malloc((size_t)reps * sizeof(double));
    die_if(!ttot, "alloc failed", comm);

    double sum_local = 0.0, sum_reduce = 0.0, sum_apply = 0.0;
    for (int r = 0; r < reps; ++r) {
        tsqr_times_t t;
        TSQR_4(A_local, mloc, n, comm, Q_local, R_final, &t);
        ttot[r] = t.t_total;
        sum_local += t.t_localqr;
        sum_reduce += t.t_reduce;
        sum_apply += t.t_apply;
    }

    double t_total_mean = mean(ttot, reps);
    double t_total_std = stddev(ttot, reps);
    double t_local_mean = sum_local / (double)reps;
    double t_red_mean = sum_reduce / (double)reps;
    double t_app_mean = sum_apply / (double)reps;

    /* correctness AFTER timing; uses last Q_local and last R_final */
    double rel_res = NAN, orth = NAN;
    int ok = compute_relres_orth(m, n, mloc, Q_local, R_final, comm, seed, &rel_res, &orth);

    if (rank == 0) {
        if (ok) {
            printf("%s,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,OK\n",
                mode, m, n, mloc, reps, warmup,
                t_total_mean, t_total_std, t_local_mean, t_red_mean, t_app_mean,
                rel_res, orth);
        }
        else {
            printf("%s,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,,,OK\n",
                mode, m, n, mloc, reps, warmup,
                t_total_mean, t_total_std, t_local_mean, t_red_mean, t_app_mean);
        }
        fflush(stdout);
    }

    free(ttot);
    if (rank == 0) free(R_final);
    free(A_local);
    free(Q_local);
}


static int streq(const char* a, const char* b) { return strcmp(a, b) == 0; }

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    die_if(size != 4, "Run with: mpirun -np 4 ./tsqr [m n] [--reps R] [--warmup W]", comm);

    int reps = 5;
    int warmup = 1;

    /* defaults for Q3 sweeps */
    int n_fixed = 20;
    int m0 = 4000;
    int stepsm = 6;
    int m_fixed = 64000;
    int n0 = 10;
    int stepsn = 6;
    int factor = 2;

    unsigned long long seed = 1234ULL;

    /* If positional (m n) are given */
    int have_single = 0;
    int single_m = 0, single_n = 0;

    /* parse args */
    for (int i = 1; i < argc; ++i) {
        if (streq(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (streq(argv[i], "--warmup") && i + 1 < argc) warmup = atoi(argv[++i]);
        else if (streq(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (argv[i][0] != '-' && i + 1 < argc && argv[i + 1][0] != '-') {
            /* treat as single m n */
            have_single = 1;
            single_m = atoi(argv[i]);
            single_n = atoi(argv[i + 1]);
            break;
        }
    }

    if (reps < 1) reps = 1;
    if (warmup < 0) warmup = 0;

    if (rank == 0) {
        printf("mode,m,n,mloc,reps,warmup,t_total_mean,t_total_std,t_localqr_mean,t_reduce_mean,t_apply_mean,rel_res,orth,status\n");
        fflush(stdout);
    }

    if (have_single) {
        run_point("single", single_m, single_n, reps, warmup, comm, seed);
        MPI_Finalize();
        return 0;
    }

    /* Sweep 1: scale in m, fixed n=20 */
    {
        int m = m0;
        for (int s = 0; s < stepsm; ++s) {
            if (m % 4 != 0) m = ((m / 4) + 1) * 4;
            if (m < n_fixed) m = ((n_fixed / 4) + 1) * 4;
            run_point("scale_m_n20", m, n_fixed, reps, warmup, comm, seed);
            if (factor == 1) m += m0;
            else m *= factor;
        }
    }

    /* Sweep 2: scale in n, fixed m=64000 */
    {
        int n = n0;
        int m = m_fixed;
        if (m % 4 != 0) m = ((m / 4) + 1) * 4;
        for (int s = 0; s < stepsn; ++s) {
            if (n > m) break;
            run_point("scale_n_m64000", m, n, reps, warmup, comm, seed);
            if (factor == 1) n += n0;
            else n *= factor;
        }
    }

    MPI_Finalize();
    return 0;
}