import pyopencl as cl
import numpy as np
from kernels import *

# OpenCL Initializations
PLATFORM = cl.get_platforms()[0]    # 
DEVICE = PLATFORM.get_devices()[0]  # The first GPU
CONTEXT = cl.Context([DEVICE])      # For managing queues, memory, programs and kernel objects
QUEUE = cl.CommandQueue(CONTEXT)    # For sending data kernels and data transfer functions to device

PROGRAM = cl.Program(CONTEXT, transpose_kernel + mat_mult_kernel + gaussian_elimination_kernel+convolve2d_kernel).build()

def pinv_opencl(matrix):
    rows, cols = matrix.shape

    # Step 1: Transpose the matrix
    transpose_matrix = np.zeros((2, 2), dtype=np.float64)
    trans_buf = cl.Buffer(CONTEXT, cl.mem_flags.WRITE_ONLY, transpose_matrix.nbytes)
    mat_buf = cl.Buffer(CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=matrix)

    PROGRAM.transpose(QUEUE, (rows, cols), None, mat_buf, np.int32(rows), np.int32(cols), trans_buf)
    cl.enqueue_copy(QUEUE, transpose_matrix, trans_buf).wait()

    # Step 2: Multiply transpose with the original matrix (A^T * A)
    product_matrix = np.zeros((cols, cols), dtype=np.float64)
    prod_buf = cl.Buffer(CONTEXT, cl.mem_flags.WRITE_ONLY, product_matrix.nbytes)

    PROGRAM.mat_mult(QUEUE, (cols, cols), None, trans_buf, mat_buf, prod_buf, np.int32(cols), np.int32(rows), np.int32(cols))
    cl.enqueue_copy(QUEUE, product_matrix, prod_buf).wait()

    # Step 3: Gaussian elimination to find the inverse of (A^T * A)
    aug_matrix = np.zeros((cols, 2 * cols), dtype=np.float64)
    for i in range(cols):
        aug_matrix[i, :cols] = product_matrix[i, :]
        aug_matrix[i, cols + i] = 1.0

    aug_matrix = np.ascontiguousarray(aug_matrix)
    aug_buf = cl.Buffer(CONTEXT, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR, hostbuf=aug_matrix)
    for i in range(cols):
        PROGRAM.normalize_row(QUEUE, (1,), None, aug_buf, np.int32(i), np.int32(cols))
        for j in range(cols):
            if i != j:
                PROGRAM.eliminate_row(QUEUE, (1,), None, aug_buf, np.int32(i), np.int32(j), np.int32(cols))
    cl.enqueue_copy(QUEUE, aug_matrix, aug_buf).wait()
    inverse_matrix = aug_matrix[:, cols:]

    pinv_matrix = np.zeros((cols, rows), dtype=np.float64)
    inv_buf = cl.Buffer(CONTEXT, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=np.ascontiguousarray(inverse_matrix))
    pinv_buf = cl.Buffer(CONTEXT, cl.mem_flags.WRITE_ONLY, pinv_matrix.nbytes)
    PROGRAM.mat_mult(QUEUE, (cols, rows), None, inv_buf, trans_buf, pinv_buf, np.int32(cols), np.int32(cols), np.int32(rows))
    cl.enqueue_copy(QUEUE, pinv_matrix, pinv_buf).wait()

    return pinv_matrix

def convolve2d_opencl(image, filter):
    print("OpenCL convolution")
    N1, N2 = image.shape
    M1, M2 = filter.shape

    # Output to store the result
    out = np.zeros_like(image, dtype=np.float64)

    image = np.array(image).astype(np.float64)
    filter = np.array(filter).astype(np.float64)

    # Create input and output buffers
    mf = cl.mem_flags
    image_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=image)
    filter_buf = cl.Buffer(CONTEXT, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=filter)
    out_buf = cl.Buffer(CONTEXT, mf.WRITE_ONLY, out.nbytes)

    # Run the kernel
    PROGRAM.convolve2d(QUEUE, out.shape, None, image_buf, np.int32(N1), np.int32(N2), filter_buf, np.int32(M1), np.int32(M2), out_buf)

    # Copy result back from GPU to host (CPU)
    cl.enqueue_copy(QUEUE, out, out_buf).wait()

    return out