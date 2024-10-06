#ifndef SYMNMF_H
#define SYMNMF_H

extern const char *GENERAL_ERROR;
extern double BETA;
extern double EPSILON;
extern int MAX_ITER;

/**
 * Calculates the squared Euclidean distance between two points.
 * @param point1 The first point.
 * @param point2 The second point.
 * @param dim The dimension of the points.
 * @return The squared Euclidean distance.
 */
double euc_dist_square(double* point1, double* point2, int dim);

/**
 * Computes the inverse square root of the diagonal degree matrix.
 * @param matrix The diagonal degree matrix.
 * @param n The size of the matrix.
 */
void get_diag_inv_square(double **matrix, int n);

/**
 * Multiplies two matrices.
 * @param rows1 The number of rows in the first matrix.
 * @param cols1 The number of columns in the first matrix.
 * @param cols2 The number of columns in the second matrix.
 * @param matrix1 The first matrix.
 * @param matrix2 The second matrix.
 * @param result The result matrix.
 */
void matrix_mult(int rows1, int cols1, int cols2, double** matrix1, double** matrix2, double** result);

/**
 * Computes the similarity matrix.
 * @param data The data points.
 * @param n The number of data points.
 * @param dim The dimension of the data points.
 * @param result The similarity matrix.
 */
void get_sym(double** data, int n, int dim, double** result);

/**
 * Computes the diagonal degree matrix.
 * @param sym_matrix The similarity matrix.
 * @param n The size of the matrix.
 * @param result The diagonal degree matrix.
 */
void get_ddg(double** sym_matrix, int n, int dim, double** result);

/**
 * Computes the normalized similarity matrix.
 * @param sym_matrix The similarity matrix.
 * @param n The size of the matrix.
 * @param result The normalized similarity matrix.
 */
void get_norm(double** sym_matrix, int n, int dim, double** result);

/**
 * Finds the dimension of the data points.
 * @param file The input file.
 * @return The dimension of the data points.
 */
int find_dimension(FILE* file);

/**
 * Splits a string into an array of doubles.
 * @param str The input string.
 * @param array The output array of doubles.
 * @param dim The dimension of the data points.
 */
void split_string_to_doubles(char* str, double* array, int dim);

/**
 * Counts the number of lines in a file.
 * @param file The input file.
 * @return The number of lines.
 */
int count_lines(FILE* file);

/**
 * Reads data points from a file into a 2D array.
 * @param file The input file.
 * @param n The number of data points.
 * @param dim The dimension of the data points.
 * @return The 2D array of data points.
 */
double** get_data_points_array(FILE* file, int n, int dim);

/**
 * Parses data from a file.
 * @param file The input file.
 * @param n The number of data points.
 * @param dim The dimension of the data points.
 * @return The 2D array of parsed data points.
 */
double **parse_data(FILE* file, int* n, int* dim);

#endif // SYMNMF_H
