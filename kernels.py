convolve2d_kernel="""__kernel void convolve2d(__global const double* in1,
                         const int N1,
                         const int N2,
                         __global const double* in2,
                         const int M1,
                         const int M2,
                         __global double* out) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < N1 && j < N2) {
        double sum = 0.0;
        for (int m = 0; m < M1; m++) {
            for (int n = 0; n < M2; n++) {
                int ii = i - (M1 / 2) + m;
                int jj = j - (M2 / 2) + n;

                // Apply boundary conditions (zero-padding)
                if (ii >= 0 && ii < N1 && jj >= 0 && jj < N2) {
                    sum += in1[ii * N2 + jj] * in2[m * M2 + n];
                }
            }
        }
        if(sum != 0){
          out[i * N2 + j] = -sum;
        }else{
          out[i * N2 + j] = sum;
        }
    }
}
"""

transpose_kernel = """__kernel void transpose(__global const double* in1,
                         const int rows,
                         const int cols,
                         __global double* out) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if(i < rows && j < cols){
      out[j * rows + i] = in1[i * cols + j];
    }
}
"""

mat_mult_kernel = """
__kernel void mat_mult(__global const double* mat1, __global const double* mat2, __global double* result, const int rows1, const int cols1, const int cols2) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < rows1 && j < cols2) {
        double sum = 0.0;
        for (int k = 0; k < cols1; k++) {
            sum += mat1[i * cols1 + k] * mat2[k * cols2 + j];
        }
        result[i * cols2 + j] = sum;
    }
}
"""

gaussian_elimination_kernel = """
__kernel void normalize_row(__global double* mat, const int row, const int n) {
    double divisor = mat[row * 2 * n + row];
    for (int j = 0; j < 2 * n; j++) {
        mat[row * 2 * n + j] /= divisor;
    }
}

__kernel void eliminate_row(__global double* mat, const int row, const int target_row, const int n) {
    double factor = mat[target_row * 2 * n + row];
    for (int j = 0; j < 2 * n; j++) {
        mat[target_row * 2 * n + j] -= factor * mat[row * 2 * n + j];
    }
}
"""
