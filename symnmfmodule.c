#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "symnmf.h"

static int convertPyArrToCArr(PyObject *, double**, int, int);
static PyObject *convertCArrToPyArr(double **, int, int);
static PyObject *symnmf(PyObject *, PyObject *);
static void freeTwoDArr(double**, int);
static void prepExit(int, double**, double**, double**, double**, double**, double**, double**, int);
void subtract_matrices(int, int, double**, double**, double**);
void transpose_matrix(int, int, double **, double **);
double frobenius_norm_square(int, int, double **);
void copyMatrix(double**, double**, int, int);

/**
 * Copies the contents of source matrix to destination matrix.
 */
void copyMatrix(double** source, double** destination, int rows, int cols) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            destination[i][j] = source[i][j];
        }
    }
}

/**
 * Frees allocated memory and handles errors if necessary.
 */
void prepExit(int n, double** matrix1, double** matrix2, double** matrix3, double** matrix4,
              double** matrix5, double** matrix6, double** matrix7, int error) {
    
    if(matrix1 != NULL) freeTwoDArr(matrix1, n);
    if(matrix2 != NULL) freeTwoDArr(matrix2, n);
    if(matrix3 != NULL) freeTwoDArr(matrix3, n);
    if(matrix4 != NULL) freeTwoDArr(matrix4, n);
    if(matrix5 != NULL) freeTwoDArr(matrix5, n);
    if(matrix6 != NULL) freeTwoDArr(matrix6, n);
    if(matrix7 != NULL) freeTwoDArr(matrix7, n);
    
    if(error == 1) {
        PyErr_SetString(PyExc_RuntimeError, GENERAL_ERROR);
    }
}

/**
 * Calculates the Frobenius norm squared of a matrix.
 */
double frobenius_norm_square(int rows, int cols, double** matrix) {
    double sum = 0.0;
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            sum += matrix[i][j] * matrix[i][j];
        }
    }
    return sum;
}

/**
 * Transposes the source matrix into the transposed matrix.
 */
void transpose_matrix(int rows, int cols, double** sourceMatrix, double** transposedMatrix) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            transposedMatrix[j][i] = sourceMatrix[i][j];
        }
    }
}

/**
 * Subtracts matrixB from matrixA and stores the result in resultMatrix.
 */
void subtract_matrices(int rows, int cols, double** matrixA, double** matrixB, double** resultMatrix) {
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            resultMatrix[i][j] = matrixA[i][j] - matrixB[i][j];
        }
    }
}

/**
 * Frees a 2D array.
 */
static void freeTwoDArr(double** arr, int lim) {
    int i;
    for (i = 0; i < lim; i++) {
        free(arr[i]);
    }
    free(arr);
}

/**
 * Converts a Python list to a C array.
 */
static int convertPyArrToCArr(PyObject* pyArr, double** cArr, int rows, int cols) {
    int outerIndex, innerIndex;
    PyObject *currSubArr, *currValue;
    for (outerIndex = 0; outerIndex < rows; outerIndex++) {
        currSubArr = PyList_GetItem(pyArr, outerIndex);
        if (currSubArr == NULL) {
            return 0;
        }
        for (innerIndex = 0; innerIndex < cols; innerIndex++) {
            currValue = PyList_GetItem(currSubArr, innerIndex);
            if (currValue == NULL) {
                return 0;
            }
            cArr[outerIndex][innerIndex] = PyFloat_AsDouble(currValue);
        }
    }
    return 1;
}

/**
 * Converts a C array to a Python list.
 */
static PyObject *convertCArrToPyArr(double **cArr, int rows, int cols) {
    int outerIndex, innerIndex;
    PyObject *pyArr = PyList_New(rows), *currSubArr, *currValue;
    for (outerIndex = 0; outerIndex < rows; outerIndex++) {
        currSubArr = PyList_New(cols);
        if (currSubArr == NULL) {
            return NULL;
        }
        for (innerIndex = 0; innerIndex < cols; innerIndex++) {
            currValue = PyFloat_FromDouble(cArr[outerIndex][innerIndex]);
            PyList_SetItem(currSubArr, innerIndex, currValue);
        }
        PyList_SetItem(pyArr, outerIndex, currSubArr);
    }
    return pyArr;
}

/**
 * Allocates memory for a 2D array of doubles.
 */
double** allocateDoublesMatrix(int rows, int cols) {
    int i;
    double** matrix = malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        return NULL;
    }

    for (i = 0; i < rows; i++) {
        matrix[i] = malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            freeTwoDArr(matrix, i);
            return NULL;
        }
    }
    return matrix;
}

/**
 * Performs the Symmetric Non-negative Matrix Factorization (SymNMF) algorithm.
 */
static PyObject *symnmf(PyObject *self, PyObject *args) {
    int i, j, n, k, iter;
    double** decomp, **norm_sim, **new_decomp, **numerator, **denominator, **trans_decomp, **squared_decomp, **diff_decomp;
    PyObject* py_decomp, *py_norm_sim;

    PyArg_ParseTuple(args, "iiOO", &n, &k, &py_decomp, &py_norm_sim);

    decomp = allocateDoublesMatrix(n, k);
    if (decomp == NULL) {
        prepExit(n, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    norm_sim = allocateDoublesMatrix(n, n);
    if (norm_sim == NULL) {
        prepExit(n, decomp, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    new_decomp = allocateDoublesMatrix(n, k);
    if (new_decomp == NULL) {
        prepExit(n, decomp, norm_sim, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    numerator = allocateDoublesMatrix(n, k);
    if (numerator == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    denominator = allocateDoublesMatrix(n, k);
    if (denominator == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, NULL, NULL, NULL, 1);
        return NULL;
    }

    trans_decomp = allocateDoublesMatrix(k, n);
    if (trans_decomp == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, NULL, NULL, 1);
        return NULL;
    }

    squared_decomp = allocateDoublesMatrix(n, n);
    if (squared_decomp == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, NULL, NULL, 1);
        freeTwoDArr(trans_decomp, k);
        return NULL;
    }

    diff_decomp = allocateDoublesMatrix(n, n);
    if (diff_decomp == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, squared_decomp, NULL, 1);
        freeTwoDArr(trans_decomp, k);
        return NULL;
    }

    if (convertPyArrToCArr(py_decomp, decomp, n, k) == 0) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, squared_decomp, diff_decomp, 1);
        freeTwoDArr(trans_decomp, k);
        return NULL;
    }

    if (convertPyArrToCArr(py_norm_sim, norm_sim, n, n) == 0) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, squared_decomp, diff_decomp, 1);
        freeTwoDArr(trans_decomp, k);
        return NULL;
    }

    for (iter = 0; iter < MAX_ITER; iter++) {
        transpose_matrix(n, k, decomp, trans_decomp);
        matrix_mult(n, n, k, norm_sim, decomp, numerator);
        matrix_mult(n, k, n, decomp, trans_decomp, squared_decomp);
        matrix_mult(n, n, k, squared_decomp, decomp, denominator);
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                new_decomp[i][j] = decomp[i][j] * ((1 - BETA) + BETA * (numerator[i][j] / denominator[i][j]));
            }
        }
        subtract_matrices(n, k, new_decomp, decomp, diff_decomp);
        if (frobenius_norm_square(n, k, diff_decomp) < EPSILON) {
            break;
        }
        copyMatrix(new_decomp, decomp, n, k);
    }

    PyObject* pyObj = convertCArrToPyArr(new_decomp, n, k);
    if (pyObj == NULL) {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, squared_decomp, diff_decomp, 1);
    } else {
        prepExit(n, decomp, norm_sim, new_decomp, numerator, denominator, squared_decomp, diff_decomp, 0);
    }
    freeTwoDArr(trans_decomp, k);
    return pyObj;
}

/**
 * Computes the similarity matrix.
 */
static PyObject *sym(PyObject *self, PyObject *args) {
    int n, dim;
    double** points_array, **sim_mat;
    PyObject* py_data;

    PyArg_ParseTuple(args, "iiO", &n, &dim, &py_data);

    points_array = allocateDoublesMatrix(n, dim);
    if (points_array == NULL) {
        prepExit(n, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    sim_mat = allocateDoublesMatrix(n, n);
    if (sim_mat == NULL) {
        prepExit(n, points_array, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    if (convertPyArrToCArr(py_data, points_array, n, dim) == 0) {
        prepExit(n, points_array, sim_mat, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    get_sym(points_array, dim, n, sim_mat);
    PyObject* pyObj = convertCArrToPyArr(sim_mat, n, n);
    if (pyObj == NULL)
        prepExit(n, points_array, sim_mat, NULL, NULL, NULL, NULL, NULL, 1);
    else
        prepExit(n, points_array, sim_mat, NULL, NULL, NULL, NULL, NULL, 0);
    return pyObj;
}

/**
 * Computes the diagonal degree matrix.
 */
static PyObject *ddg(PyObject *self, PyObject *args) {
    int n, dim;
    double** points_array, **ddg_mat;
    PyObject* py_data;

    PyArg_ParseTuple(args, "iiO", &n, &dim, &py_data);

    points_array = allocateDoublesMatrix(n, dim);
    if (points_array == NULL) {
        prepExit(n, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    ddg_mat = allocateDoublesMatrix(n, n);
    if (ddg_mat == NULL) {
        prepExit(n, points_array, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    if (convertPyArrToCArr(py_data, points_array, n, dim) == 0) {
        prepExit(n, points_array, ddg_mat, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    get_ddg(points_array, dim, n, ddg_mat);
    PyObject* pyObj = convertCArrToPyArr(ddg_mat, n, n);
    if (pyObj == NULL)
        prepExit(n, points_array, ddg_mat, NULL, NULL, NULL, NULL, NULL, 1);
    else
        prepExit(n, points_array, ddg_mat, NULL, NULL, NULL, NULL, NULL, 0);

    return pyObj;
}

/**
 * Computes the normalized similarity matrix.
 */
static PyObject *norm(PyObject *self, PyObject *args) {
    int n, dim;
    double** points_array, **norm_mat;
    PyObject* py_data;

    PyArg_ParseTuple(args, "iiO", &n, &dim, &py_data);

    points_array = allocateDoublesMatrix(n, dim);
    if (points_array == NULL) {
        prepExit(n, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    norm_mat = allocateDoublesMatrix(n, n);
    if (norm_mat == NULL) {
        prepExit(n, points_array, NULL, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    if (convertPyArrToCArr(py_data, points_array, n, dim) == 0) {
        prepExit(n, points_array, norm_mat, NULL, NULL, NULL, NULL, NULL, 1);
        return NULL;
    }

    get_norm(points_array, dim, n, norm_mat);
    PyObject* pyObj = convertCArrToPyArr(norm_mat, n, n);
    if (pyObj == NULL)
        prepExit(n, points_array, norm_mat, NULL, NULL, NULL, NULL, NULL, 1);
    else
        prepExit(n, points_array, norm_mat, NULL, NULL, NULL, NULL, NULL, 0);

    return pyObj;
}

/* Create wrapper functions for the symnmf functions */
static PyMethodDef symnmf_methods[] = {
    {"symnmf", (PyCFunction)symnmf, METH_VARARGS, 
        PyDoc_STR("The symnmf function receives as input the following arguments:"
                  "n - a Python Integer representing the rows of the decomp matrix"
                  "k - a Python Integer representing the columns of the decomp matrix"
                  "py_decomp - a Python Object representing the decomp matrix"
                  "py_norm_sim - a Python Object representing the normalized similarity matrix")},
    {"ddg", (PyCFunction)ddg, METH_VARARGS, 
        PyDoc_STR("The ddg function receives as input the following arguments:"
                  "n - a Python Integer representing the rows of the data"
                  "dim - a Python Integer representing the columns of the data"
                  "py_data - a Python Object representing the data")},
    {"sym", (PyCFunction)sym, METH_VARARGS, 
        PyDoc_STR("The sym function receives as input the following arguments:"
                  "n - a Python Integer representing the rows of the data"
                  "dim - a Python Integer representing the columns of the data"
                  "py_data - a Python Object representing the data")},
    {"norm", (PyCFunction)norm, METH_VARARGS, 
        PyDoc_STR("The norm function receives as input the following arguments:"
                  "n - a Python Integer representing the rows of the data"
                  "dim - a Python Integer representing the columns of the data"
                  "py_data - a Python Object representing the data")},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "mysymnmf", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symnmf_methods /* the PyMethodDef array from before containing the methods of the extension */
};

PyMODINIT_FUNC PyInit_mysymnmf(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        PyErr_SetString(PyExc_RuntimeError, GENERAL_ERROR);
        return NULL;
    }
    return m;
}
