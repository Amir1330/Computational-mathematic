def determinant_2x2(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def determinant_3x3(matrix):
    return (matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
            matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
            matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]))

def determinant_4x4(matrix):
    det = 0
    for i in range(4):
        submatrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        sign = (-1) ** i
        det += sign * matrix[0][i] * determinant_3x3(submatrix)
    return det

def cramer_rule(A, b):
    D = determinant_4x4(A)
    if D == 0:
        print("Cramer's rule cannot be applied.")
        return None

    A1 = [row[:] for row in A]
    A2 = [row[:] for row in A]
    A3 = [row[:] for row in A]
    A4 = [row[:] for row in A]

    for i in range(4):
        A1[i][0] = b[i]
        A2[i][1] = b[i]
        A3[i][2] = b[i]
        A4[i][3] = b[i]

    D1 = determinant_4x4(A1)
    D2 = determinant_4x4(A2)
    D3 = determinant_4x4(A3)
    D4 = determinant_4x4(A4)

    x1 = D1 / D
    x2 = D2 / D
    x3 = D3 / D
    x4 = D4 / D

    return x1, x2, x3, x4

A = [
    [3, -5, 47, 20],
    [11, 16, 17, 10],
    [56, 22, 11, -18],
    [17, 66, -12, 7]
]

b = [18, 26, 34, 82]

solution = cramer_rule(A, b)
if solution is not None:
    x1, x2, x3, x4 = solution
    print(f"x1 = {x1} \nx2 = {x2} \nx3 = {x3} \nx4 = {x4}")
