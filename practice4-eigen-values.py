import cmath



def multiply_matrices(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        print("incorect input")
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

def multiply_matrix_vector(matrix, vector):
    if len(matrix[0]) != len(vector):
        print("Incorrect input: The number of columns in the matrix must match the number of rows in the vector.")
        return None

    result = []
    for i in range(len(matrix)):
        element_sum = 0
        for j in range(len(vector)):
            element_sum += matrix[i][j] * vector[j]
        result.append(element_sum)

    return result

def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def determinant_3x3(matrix):
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

def multiply_matrix_by_scalar(matrix, scalar):
    new_matrix = []
    for row in matrix:
        new_row = []
        for element in row:
            new_row.append(scalar * element)
        new_matrix.append(new_row)
    return new_matrix

def adjugate_2x2(matrix):
    return [
        [matrix[1][1], -matrix[0][1]],
        [-matrix[1][0], matrix[0][0]]
    ]

def inverse_2x2(matrix):
    det = determinant_2x2(matrix)
    if det == 0:
        print("Matrix is not invertible")
    adj = adjugate_2x2(matrix)
    inv_matrix = multiply_matrix_by_scalar(adj, 1 / det)
    return inv_matrix

def characteristic_polynomial_2x2(matrix):
    a = 1
    b = -(matrix[0][0] + matrix[1][1])
    c = determinant_2x2(matrix)
    return a, b, c
    #  λ^2 − (a + d) * λ + (ad − bc)

def find_eigenvalues_2x2(matrix):
    a, b, c = characteristic_polynomial_2x2(matrix)
    discriminant = cmath.sqrt(b**2 - 4*a*c)
    eigenvalue1 = (-b + discriminant) / (2*a)
    eigenvalue2 = (-b - discriminant) / (2*a)
    return eigenvalue1, eigenvalue2


def characteristic_polynomial_3x3(matrix):
    a = -1
    b = matrix[0][0] + matrix[1][1] + matrix[2][2]
    c = (matrix[0][0]*matrix[1][1] + matrix[0][0]*matrix[2][2] + matrix[1][1]*matrix[2][2] -
    matrix[0][1]*matrix[1][0] - matrix[0][2]*matrix[2][0] - matrix[1][2]*matrix[2][1])
    d = determinant_3x3(matrix)
    return a, b, c, d

def solve_cubic(a, b, c, d):
    # aλ^3 + bλ^2 + cλ + d = 0
    p = -b / (3 * a)
    q = p**3 + (b*c - 3*a*d) / (6 * a**2)
    r = c / (3 * a)
    discriminant = q**2 + (r - p**2)**3

    s = cmath.sqrt(q**2 + (r - p**2)**3)
    t1 = (q + s)**(1/3)
    t2 = (q - s)**(1/3)

    root1 = t1 + t2 + p
    root2 = -(t1 + t2) / 2 + p + cmath.sqrt(3) * (t1 - t2) / 2 * 1j
    root3 = -(t1 + t2) / 2 + p - cmath.sqrt(3) * (t1 - t2) / 2 * 1j
    return root1, root2, root3


def find_eigenvalues_3x3(matrix):
    a, b, c, d = characteristic_polynomial_3x3(matrix)
    eigenvalues = solve_cubic(a, b, c, d)
    return eigenvalues


def get_minor(matrix, row, col):
    minor = []
    for i in range(len(matrix)):
        if i != row:
            minor_row = []

            for j in range(len(matrix[i])):
                if j != col:
                    minor_row.append(matrix[i][j])
            minor.append(minor_row)

    return minor

def matrix_of_minors(matrix):
    minors = []
    for i in range(3):
        minors_row = []
        for j in range(3):
            minor = get_minor(matrix, i, j)
            minors_row.append(determinant_2x2(minor))
        minors.append(minors_row)
    return minors

def cofactor_matrix(matrix):
    return [
        [matrix[0][0], -matrix[0][1],  matrix[0][2]],
        [-matrix[1][0], matrix[1][1], -matrix[1][2]],
        [matrix[2][0], -matrix[2][1],  matrix[2][2]]
    ]

def inverse_3x3(matrix):
    minorM = matrix_of_minors(matrix)
    cofacM = cofactor_matrix(minorM)
    det = determinant_3x3(matrix)
    result = multiply_matrix_by_scalar(cofacM, 1/det)
    return result



# MAIN

matrix = [[3, 4],
          [2, 1]]

eigenvalues = find_eigenvalues_2x2(matrix)
print("Eigenvalues of 2x2 matrix:", eigenvalues)
print('\n')

inverse2x2 = inverse_2x2(matrix)
if inverse2x2:
    print("Inverse matrix of 2x2:")
    for row in inverse2x2:
        print(row)
print('\n')
# print("check for invers A * A^(-1) = I2", multiply_matrices(matrix, inverse_2x2(matrix)))

matrix3x3 = [
    [2, 4, 2],
    [3, 1, 1],
    [1, 0, 1]
]

eigenvalues = find_eigenvalues_3x3(matrix3x3)
print(f"Eigenvalues of 3x3: {eigenvalues}")
print('\n')

inverse = inverse_3x3(matrix3x3)
if inverse:
    print("Inverse matrix:")
    for row in inverse:
        print(row)
print('\n')
