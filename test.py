def adjugate_2x2(matrix):
    return [
        [matrix[1][1], -matrix[0][1]],
        [-matrix[1][0], matrix[0][0]]
    ]

def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def multiply_matrix_by_scalar(matrix, scalar):
    new_matrix = []
    for row in matrix:
        new_row = []
        for element in row:
            new_row.append(scalar * element)
        new_matrix.append(new_row)
    return new_matrix

def inverse_2x2(matrix):
    det = determinant_2x2(matrix)
    if det == 0:
        raise ValueError("Matrix is not invertible")
    adj = adjugate_2x2(matrix)
    inv_matrix = multiply_matrix_by_scalar(adj, 1 / det)
    return inv_matrix

def multiply_matrices(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        print("Incorrect input")
        return None
    result = []
    for i in range(len(matrix1)):
        result_row = []
        for j in range(len(matrix2[0])):
            element_sum = 0
            for k in range(len(matrix2)):
                element_sum += matrix1[i][k] * matrix2[k][j]
            result_row.append(element_sum)
        result.append(result_row)
    return result

# Example usage
matrix = [[1, 2], [3, 4]]
inverse_matrix = inverse_2x2(matrix)
print("Inverse of the matrix is:", inverse_matrix)

# Verify by multiplying the matrix by its inverse
identity_matrix = multiply_matrices(matrix, inverse_matrix)
print("Product of matrix and its inverse is:", identity_matrix)
