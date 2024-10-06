import sys
import os
import numpy as np
import pandas as pd
import mysymnmf

MAX_ITER = 300
EPSILON = 1e-4
GENERAL_ERROR = "An Error Has Occurred"

def print_matrix(matrix):
    """Prints a matrix in a formatted way, with each row on a new line and columns separated by commas."""
    my_string = ''
    for row in matrix:
        my_string += ','.join(map(lambda x: "{:.4f}".format(x), row)) + '\n'
    my_string = my_string.rstrip('\n')
    print(my_string)

def get_symnmf(n, k, dim, data_points):
    """Performs step 1.4 in the project instructions."""
    np.random.seed(0)
    normal_sim = mysymnmf.norm(n, dim, data_points)  # Get W (the normalized similarity matrix) from C
    norm_sim_mean = np.mean(normal_sim)
    upper_bound = 2 * np.sqrt(norm_sim_mean / k)  # 2 * sqrt(m/k)

    # 1.4.1 - Initialize H
    decomp = np.random.uniform(0, upper_bound, size=(n, k))
    decomp = decomp.tolist()
    # Call the symnmf method
    return mysymnmf.symnmf(n, k, decomp, normal_sim)

def read_input_file(file_name):
    """Reads the input file into a pandas DataFrame."""
    if not os.path.exists(file_name):
        print(GENERAL_ERROR)
        sys.exit(1)
    try:
        return pd.read_csv(file_name, header=None)
    except Exception:
        print(GENERAL_ERROR)
        sys.exit(1)

def validate_k(k, n):
    """Validates the number of clusters."""
    if k <= 1 or k >= n:
        print(GENERAL_ERROR)
        sys.exit(1)

def main():
    if len(sys.argv) != 4:
        print(GENERAL_ERROR)
        sys.exit(1)

    try:
        k = int(sys.argv[1])
    except ValueError:
        print(GENERAL_ERROR)
        sys.exit(1)

    goal = sys.argv[2]
    file_name = sys.argv[3]

    file_df = read_input_file(file_name)
    validate_k(k, file_df.shape[0])

    dim = file_df.shape[1]
    data_points = file_df.values.tolist()
    n = file_df.shape[0]

    matrix = None
    try:
        if goal == 'sym':
            # Call the main C sym method
            matrix = mysymnmf.sym(n, dim, data_points)
        elif goal == 'ddg':
            # Call the main C ddg method
            matrix = mysymnmf.ddg(n, dim, data_points)
        elif goal == 'norm':
            # Call the main C norm method
            matrix = mysymnmf.norm(n, dim, data_points)
        elif goal == 'symnmf':
            matrix = get_symnmf(n, k, dim, data_points)
        else:
            raise Exception(GENERAL_ERROR)

        print_matrix(matrix)
    except RuntimeError:
        print(GENERAL_ERROR)
        sys.exit(1)

if __name__ == "__main__":
    main()
