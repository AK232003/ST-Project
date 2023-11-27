import numpy as np
import math
from decimal import Decimal

class Number:
    def __init__(self, num, precision):
        self.num = num
        self.precision = precision
        
def make_equal_length(num1, num2):
    if(num2 == None or num1 == None):
        # print("ERROR: Cannot make it of Equal Length")
        return -1
    len1 = len(num1.num)
    len2 = len(num2.num)
    # print(len1, len2)
    if len1 < len2:
        for i  in range(len2 - len1):
            num1.num.insert(0, 0)
        # num1.num = [0] * (len2 - len1) + num1.num
    elif len1 > len2:
        for i  in range(len1 - len2):
            num2.num.insert(0, 0)

    if len(num1.num) == len(num2.num):
        return len(num1.num)
    
    
def padding_to_precision(num1, num2):
    if num1.precision < num2.precision:
        for i in range(num2.precision - num1.precision):
            num1.num.append(0)
        num1.precision = num2.precision
    elif num1.precision > num2.precision:
        for i in range(num1.precision - num2.precision):
            num2.num.append(0)
        num2.precision = num1.precision

def is_first_smaller(num1, num2):
    len1 = len(num1.num)
    len2 = len(num2.num)

    if len1 < len2:
        return True
    elif len2 < len1:
        return False
    else:
        for i in range(len1):
            if num1.num[i] < num2.num[i]:
                return True
            elif num1.num[i] > num2.num[i]:
                return False
        return False

def make_shifting(tobe_shifted, shift_count):
    after_shift = Number([], tobe_shifted.precision)
    for i in range(len(tobe_shifted.num)):
        after_shift.num.append(tobe_shifted.num[i])
        
    for _ in range(shift_count):
        after_shift.num.append(0)

    return after_shift

def remove_padding_zeros(dividend, divisor):
    while dividend.num[0] == 0:
        dividend.num.pop(0)
    
    while divisor.num[0] == 0:
        divisor.num.pop(0)

def truncate(x):
    while x.num[0] == 0 and len(x.num) > 1:
        x.num.pop(0)
        
def is_same(prev_root_X, root_X, p, B):
    if prev_root_X.precision == p and root_X.precision == p:
        if len(prev_root_X.num) == len(root_X.num):
            for i in range(len(prev_root_X.num) - 1, -1, -1):
                if prev_root_X.num[i] != root_X.num[i]:
                    return False
            return True
        else:
            return False
    else:
        return False

def padding(x, p):
    for i in range(p):
        x.num.append(0)

def large_integer_addition(B, num1, num2):
    padding_to_precision(num1, num2)
    # print(num1.num, num2.num)
    len_max = make_equal_length(num1, num2)

    result = Number([], max(num1.precision, num2.precision))

    carry = 0
    temp = []

    for i in range(len_max - 1, -1, -1):
        sum_val = num1.num[i] + num2.num[i] + carry
        temp.append(sum_val % B)
        carry = sum_val // B

    if carry > 0:
        temp.append(carry)
        temp.reverse()
        for i in range(len_max + 1):
            result.num.append(temp[i])
        # result.num = temp
    else:
        temp.reverse()
        for i in range(len_max):
            result.num.append(temp[i])
        # result.num = temp

    return result

def large_integer_subtraction(B, num1, num2):
    padding_to_precision(num1, num2)

    len_max = make_equal_length(num1, num2) # NUM2, NUM1 IT

    result = Number([], max(num1.precision, num2.precision))

    carry = 0
    temp = []

    if is_first_smaller(num1, num2):
        for i in range(len_max - 1, -1, -1):
            diff = num2.num[i] - num1.num[i] - carry
            if diff < 0:
                diff += B
                carry = 1
            else:
                carry = 0

            temp.append(diff)

        temp[len_max - 1] = temp[len_max - 1] * -1
    else:
        for i in range(len_max - 1, -1, -1):
            diff = num1.num[i] - num2.num[i] - carry
            if diff < 0:
                diff += B
                carry = 1
            else:
                carry = 0

            temp.append(diff)

    temp.reverse()
    for i in range(len_max):
        result.num.append(temp[i])

    return result

def large_integer_multiplication(B, num1, num2):
    padding_to_precision(num1, num2)
    len_max = make_equal_length(num1, num2)
    if len_max == 1:
        single_bit_multiplication = Number([(num1.num[0] * num2.num[0]) // B, (num1.num[0] * num2.num[0]) % B], num1.precision + num2.precision)
        return single_bit_multiplication

    lh = len_max // 2
    rh = len_max - lh
    num1L = Number([], num1.precision)
    num1R = Number([], num1.precision)
    num2L = Number([], num2.precision)
    num2R = Number([], num2.precision)

    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    
    for i in range(0, lh):
        temp1.append(num1.num[i])
        temp2.append(num2.num[i])
        
    for i in range(lh, len_max):
        temp3.append(num1.num[i])
        temp4.append(num2.num[i])
    
    num1L.num = temp1
    num2L.num = temp2
    num1R.num = temp3
    num2R.num = temp4
    
    P1 = large_integer_multiplication(B, num1L, num2L) # DUMY STRUCT IT
    P2 = large_integer_multiplication(B, num1R, num2R)    
    P3 = large_integer_multiplication(B, large_integer_addition(B, num1L, num1R), large_integer_addition(B, num2L, num2R))
    P4 = large_integer_subtraction(B, large_integer_subtraction(B, P3, P1), P2)
    P5 = make_shifting(P1, 2 * rh)
    P6 = make_shifting(P4, rh)
    P7 = large_integer_addition(B, P5, P2)
    if(P6.num[0] < 0):
        P6.num[0] = P6.num[0]*-1
        P8 = large_integer_subtraction(B, P7, P6)
    else:
        P8 = large_integer_addition(B, P7, P6)

    return P8

def large_integer_division(B, num1, num2, quotient, remainder):
    flag = False
    k = len(num1.num)
    l = len(num2.num)
    count = 0

    if k < l:
        quotient.num.append(0)
        for i in range(k):
            remainder.num.append(num1.num[i])
        return
    elif l < k:
        flag = True
    else:
        for i in range(k):
            if num1.num[i] > num2.num[i]:
                flag = True
                break
            elif num2.num[i] > num1.num[i]:
                quotient.num.append(0)
                for j in range(k):
                    remainder.num.append(num1.num[j])
                return
        
    if not flag:
        quotient.num.append(1)
        remainder.num.append(0)
        return

    k = len(num1.num)
    l = len(num2.num)

    num1.num.reverse()
    num2.num.reverse()

    for i in range(k):
        remainder.num.append(num1.num[i])
    
    remainder.num.append(0)
    for i in range(k - l + 1):
        quotient.num.append(0)

    for i in range(k - l, -1, -1):
        quotient.num[i] = (remainder.num[i + l] * B + remainder.num[i + l - 1]) // num2.num[l - 1]
        if quotient.num[i] >= B:
            quotient.num[i] = B - 1

        carry = 0
        count = 0

        for j in range(l):
            tmp = remainder.num[i + j] - quotient.num[i] * num2.num[j] + carry
            carry = tmp // B
            remainder.num[i + j] = tmp % B
            if remainder.num[i + j] < 0:
                remainder.num[i + j] += B
                carry -= 1

        remainder.num[i + l] += carry

        while remainder.num[i + l] < 0:
            carry = 0
            for j in range(l):
                tmp = remainder.num[i + j] + num2.num[j] + carry
                carry = tmp // B
                remainder.num[i + j] = tmp % B
                if remainder.num[i + j] < 0:
                    remainder.num[i + j] += B
                    carry -= 1

            remainder.num[i + l] += carry
            quotient.num[i] -= 1

    # print(carry)
    quotient.num.reverse()
    remainder.num.reverse()

    for i in range(len(quotient.num)):
        if quotient.num[i] != 0:
            truncate(quotient)
            break

def large_real_division(B, num1, num2, quotient, remainder, p):
    padding_to_precision(num1, num2)
    padding(num1, p)
    large_integer_division(B, num1, num2, quotient, remainder)
    quotient.precision = p

def vector_norm(x):
    """Compute the L2 norm of a vector."""
    return Decimal(float(np.dot(x, x))).sqrt()

def qr_decomposition(A):
    """
    Compute the QR decomposition of a matrix A using the Gram-Schmidt algorithm.

    Parameters:
    A (numpy.ndarray): The matrix to decompose.

    Returns:
    (numpy.ndarray, numpy.ndarray): A tuple of Q and R matrices that represent the QR decomposition of A, where:
        Q (numpy.ndarray): The orthogonal matrix Q.
        R (numpy.ndarray): The upper triangular matrix R.
    """
    # Get the shape of the input matrix
    m, n = A.shape

    # Initialize the matrices
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Perform the Gram-Schmidt orthogonalization
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        # R[j, j] = np.linalg.norm(v)
        R[j, j] = vector_norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R # Q+1, R+1 OUTPUT SAVE IT

def rank_of_matrix(mat):
    """
    This function calculates the rank of a matrix 'mat' using Gaussian elimination method.
    It returns the rank of the matrix.
    """
    # Define the dimensions of the matrix
    m = len(mat)
    n = len(mat[0])

    rank = min(m, n)

    # Perform Gaussian elimination
    for row in range(rank):
        # Check if the diagonal element is not zero
        if mat[row][row] != 0:
            for col in range(row + 1, m):
                # Calculate the factor by which to multiply the current row
                # to eliminate the non-zero element in the current column
                factor = mat[col][row] / mat[row][row]
                for i in range(row, n):
                    # Update the current row by subtracting the product of the factor
                    # and the corresponding element in the row being eliminated from it
                    mat[col][i] -= factor * mat[row][i]
        else:
            # If the diagonal element is zero, look for a non-zero element below it
            # and swap the rows if necessary
            reduce_rank = True
            for i in range(row + 1, m):
                if mat[i][row] != 0:
                    mat[row], mat[i] = mat[i], mat[row]
                    reduce_rank = False
                    break
            if reduce_rank:
                rank -= 1
                for i in range(row, m):
                    mat[i][row] = mat[i][rank]

    return rank

def eig(A):
    """
    Compute the eigenvalues and eigenvectors of a matrix A using the power iteration method.

    Parameters:
    A (numpy.ndarray): The matrix to compute eigenvalues and eigenvectors.

    Returns:
    (eigvals, eigvecs): A tuple of arrays that represent the eigenvalues and eigenvectors of A, where:
        eigvals (numpy.ndarray): The eigenvalues of A.
        eigvecs (numpy.ndarray): The eigenvectors of A.
    """
    # set the number of iterations and tolerance level
    max_iter = 100
    tol = 1e-6

    # initialize the eigenvectors
    m, n = A.shape
    eigvecs = np.random.randn(n, n)

    # compute the largest eigenvalue and eigenvector
    for i in range(max_iter):
        # compute the new eigenvector
        eigvecs_new = A @ eigvecs
        # eigvecs_new, _ = np.linalg.qr(eigvecs_new)
        eigvecs_new, _ = qr_decomposition(eigvecs_new)
        if np.allclose(eigvecs_new, eigvecs, rtol=tol):
            break
        eigvecs = eigvecs_new

    # compute the eigenvalues
    eigvals = np.diag(eigvecs.T @ A @ eigvecs)

    return eigvals, eigvecs

def SVD(A):
    """
    Compute Singular Value Decomposition of matrix A using NumPy.

    Args:
        A: numpy.array, matrix to be decomposed

    Returns:
        U: numpy.array, matrix containing left singular vectors
        s: numpy.array, array containing singular values
        V_T: numpy.array, matrix containing right singular vectors (transposed)
    """
    # Compute the eigenvectors and eigenvalues of A*At or At*A, whichever is smaller
    if A.shape[0] < A.shape[1]:
        S = A @ A.T
        k = rank_of_matrix(S.copy())
    else:
        S = A.T @ A
        k = rank_of_matrix(S.copy())
        
    eigvals, eigvecs = eig(S)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:,sorted_indices]

    # Compute the singular values and their reciprocals
    s = np.sqrt(eigvals)
    s = s[:k]
    s_inv = np.zeros_like(A.T)
    np.fill_diagonal(s_inv, 1.0 / s)

    # Compute the left and right singular vectors
    if(A.shape[0] > A.shape[1]):
        U = np.dot(A, np.dot(eigvecs, s_inv))
        V_T = eigvecs.T
        if(len(s) != V_T.shape[0]): V_T = V_T[:len(s) - V_T.shape[0], :] 

    else:
        U = eigvecs
        V_T = np.dot(s_inv, np.dot(U.T, A))
        if(len(s) != U.shape[1]): U = U[:, :len(s) - U.shape[1]]

    sigma = np.zeros([U.shape[1], V_T.shape[0]])
    for i in range(len(s)):
        sigma[i, i] = s[i]

    return U, s, sigma, V_T

def ReducedSVD(A, threshold=0, to_remove=0):
    U, s, sigma, V_trans = SVD(A)
    # While converting to python code we will convert into GUI asking-
    #       - Removal based on:-
    #       - 1. Hyper parameter
    #       - 2. Threshold

    # Removal based on hyper parameter
    if (to_remove < len(s) and to_remove > 0):
        s = s[:-to_remove]
        # print(s)
        U = U[:, :-to_remove]
        V_trans = V_trans[:-to_remove, :]
        sigma = sigma[:-to_remove, :-to_remove]

    elif (to_remove < 0):
        # print("The number of eigen values to be romved is Invalid!!")
        exit()

    # Removal based on threshold
    if (threshold < s[0] and threshold > 0):
        # print("HERE", s[0], threshold)
        s = s[s >= threshold]
        U = U[:, :len(s)]
        V_trans = V_trans[:len(s), :]
        sigma = sigma[:len(s), :len(s)]

    elif (threshold < 0):
        # print("Invalid threshold value!!")
        exit()

    return U, s, sigma, V_trans