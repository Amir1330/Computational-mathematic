def forward_elimination(matrix):
    n = len(matrix)
    for i in range(n):
        if matrix[i][i] == 0:
            for k in range(i + 1, n):
                if matrix[k][i] != 0:
                    matrix[i], matrix[k] = matrix[k], matrix[i]
                    break

        for j in range(i + 1, n):
            ratio = matrix[j][i] / matrix[i][i]
            for k in range(len(matrix[0])):
                matrix[j][k] -= ratio * matrix[i][k]

def back_substitution(matrix):
    n = len(matrix)
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = matrix[i][-1]
        for j in range(i+1, n):
            x[i] -= matrix[i][j] * x[j]
        x[i] /= matrix[i][i]
    return x

def gaussian_elimination(matrix):
    forward_elimination(matrix)
    return back_substitution(matrix)

matrix = [
    [3, -5, 47, 20, 18],
    [11, 16, 17, 10, 26],
    [56, 22, 11, -18, 34],
    [17, 66, -12, 7, 82]
]

solution = gaussian_elimination(matrix)

print(f"x1 = {solution[0]} \nx2 = {solution[1]} \nx3 = {solution[2]} \nx4 = {solution[3]}")
