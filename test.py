from unittest import TestCase
from NumberTheory import *
class Number:
    def __init__(self, num, precision):
        self.num = num
        self.precision = precision
        
class Test(TestCase):

    def test_make_equal_len1(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([4, 5, 6, 7], 0)
        self.assertEqual(make_equal_length(num1, num2), 4)

    def test_make_equal_len2(self):
        num1 = Number([1, 2, 3, 4, 5], 4)
        num2 = Number([6, 7, 8], 3)
        self.assertEqual(make_equal_length(num1, num2), 5)

    def test_make_equal_len2(self):
        num1 = Number([1, 2, 3, 4, 5], 4)
        num2 = None
        self.assertEqual(make_equal_length(num1, num2), -1)

    def test_smaller1(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([4, 5, 6, 7], 0)
        result = is_first_smaller(num1, num2)
        self.assertTrue(result)

    def test_smaller2(self):
        num1 = Number([0], 0)
        num2 = Number([6, 7, 8], 0)
        result = is_first_smaller(num1, num2)
        self.assertTrue(result)

    def test_smaller3(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([4, 5, 6], 0)
        result = is_first_smaller(num1, num2)
        self.assertTrue(result)

    def test_smaller4(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([1, 2, 3], 0)
        result = is_first_smaller(num1, num2)
        self.assertTrue(result == False)

    def test_shift1(self):
        tobe_shifted1 = Number([1, 2, 3], 0)
        result1 = make_shifting(tobe_shifted1, 2)
        self.assertEquals(result1.num, [1, 2, 3, 0, 0])

    def test_shift2(self):
        tobe_shifted2 = Number([], 0)
        result2 = make_shifting(tobe_shifted2, 3)
        self.assertEquals(result2.num, [0, 0, 0])

    def test_issame1(self):
        prev_root_X1 = Number([1, 2, 3], 4)
        root_X1 = Number([1, 2, 3], 4)
        result1 = is_same(prev_root_X1, root_X1, 4, 10)
        self.assertTrue(result1)
    
    def test_issame2(self):
        prev_root_X3 = Number([1, 2, 3], 4)
        root_X3 = Number([1, 2, 3, 4], 4)
        result3 = is_same(prev_root_X3, root_X3, 4, 10)
        self.assertTrue(result3 == False)

    def test_add1(self):
        num1_case1 = Number([1, 2, 3], 0)
        num2_case1 = Number([4, 5, 6], 0)
        result_case1 = large_integer_addition(10, num1_case1, num2_case1)
        self.assertEquals(result_case1.num, [5, 7, 9])

    def test_add2(self):
        num1_case2 = Number([9, 9, 9], 0)
        num2_case2 = Number([1], 0)
        result_case2 = large_integer_addition(10, num1_case2, num2_case2)
        self.assertEquals(result_case2.num, [1, 0, 0, 0])

    def test_add4(self):
        num1_case2 = Number([9, 9, 9], 3)
        num2_case2 = Number([1], 2)
        result_case2 = large_integer_addition(10, num1_case2, num2_case2)
        self.assertEquals(result_case2.num, [1, 0, 0, 9])

    def test_add3(self):
        num1_case4 = Number([1, 2, 3], 2)
        num2_case4 = Number([4, 5, 6], 3)
        result_case4 = large_integer_addition(10, num1_case4, num2_case4)
        self.assertEquals(result_case4.num, [1, 6, 8, 6])

    def test_sub1(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([4, 5, 6], 0)
        result1 = large_integer_subtraction(10, num1, num2)
        self.assertEquals(result1.num, [-3, 3, 3])

    def test_sub2(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([4, 5, 6], 0)
        result1 = large_integer_subtraction(10, num2, num1)
        self.assertEquals(result1.num, [3, 3, 3])

    def test_sub3(self):
        num1 = Number([1, 2, 3], 0)
        num2 = Number([1, 2, 3], 0)
        result1 = large_integer_subtraction(10, num2, num1)
        self.assertEquals(result1.num, [0, 0, 0])

    def test_sub4(self):
        num5 = Number([1, 2, 3], 2)
        num6 = Number([3, 3, 5], 3)
        result3 = large_integer_subtraction(10, num5, num6)
        self.assertEquals(result3.num, [0, 8, 9, 5])

    def test_largemul1(self):
        num1_case1 = Number([5], 0)
        num2_case1 = Number([3], 0)
        result_case1 = large_integer_multiplication(10, num1_case1, num2_case1)
        self.assertEquals(result_case1.num, [1, 5])

    def test_largemul2(self):
        # Test Case 2: Equal length multiplication
        num1_case2 = Number([1, 2, 3], 0)
        num2_case2 = Number([4, 5, 6], 0)
        result_case2 = large_integer_multiplication(10, num1_case2, num2_case2)
        self.assertEquals(result_case2.num, [0, 0, 0, 5, 6, 0, 8, 8])

    def test_largemul3(self):
        num1_case2 = Number([6, 8], 2)
        num2_case2 = Number([7, 8, 9], 3)
        result_case2 = large_integer_multiplication(10, num1_case2, num2_case2)
        self.assertEquals(result_case2.num, [0, 0, 0, 0, 5, 3, 6, 5, 2, 0])

    def test_largemul4(self):
        num1_case2 = Number([0, 0], 2)
        num2_case2 = Number([7, 8, 9], 3)
        result_case2 = large_integer_multiplication(10, num1_case2, num2_case2)
        self.assertEquals(result_case2.num, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_largediv1(self):
        num1_case6 = Number([4, 3, 7, 5], 0)
        num2_case6 = Number([2, 1], 0)
        quotient_case6 = Number([], 0)
        remainder_case6 = Number([], 0)
        large_integer_division(16, num1_case6, num2_case6, quotient_case6, remainder_case6)
        self.assertTrue(quotient_case6.num == [2, 0, 11] and remainder_case6.num == [0, 0, 0, 0, 10]) 
        # self.assertTrue(quotient_case1.num == [0] and remainder_case1.num == [1, 2, 3])
    
    def test_largediv2(self):
        num1_case1 = Number([1, 2, 3], 0)
        num2_case1 = Number([4, 5, 6, 7], 0)
        quotient_case1 = Number([], 0)
        remainder_case1 = Number([], 0)
        large_integer_division(10, num2_case1, num1_case1, quotient_case1, remainder_case1)
        self.assertTrue(quotient_case1.num == [3, 7] and remainder_case1.num == [0, 0, 0, 1, 6])

    def test_largediv3(self):
        num1_case1 = Number([1, 2, 3], 0)
        num2_case1 = Number([1, 2, 3], 0)
        quotient_case1 = Number([], 0)
        remainder_case1 = Number([], 0)
        large_integer_division(10, num1_case1, num2_case1, quotient_case1, remainder_case1)
        self.assertTrue(quotient_case1.num == [1] and remainder_case1.num == [0])

    def test_largerealdiv1(self):
        num1_case6 = Number([4, 3, 2], 2)
        num2_case6 = Number([2, 1, 3], 2)
        quotient_case6 = Number([], 0)
        remainder_case6 = Number([], 0)
        large_real_division(16, num1_case6, num2_case6, quotient_case6, remainder_case6, 3)
        self.assertTrue(quotient_case6.num == [2, 0, 5, 12] and remainder_case6.num == [0, 0, 0, 0, 1, 2, 12])

    def test_largerealdiv2(self):
        num1_case6 = Number([4, 3, 5], 2)
        num2_case6 = Number([2, 1], 1)
        quotient_case6 = Number([], 0)
        remainder_case6 = Number([], 0)
        large_real_division(16, num1_case6, num2_case6, quotient_case6, remainder_case6, 3)
        self.assertTrue(quotient_case6.num == [2, 0, 10, 2] and remainder_case6.num == [0, 0, 0, 0, 1, 14, 0])

    def test_largerealdiv3(self):
        num1_case6 = Number([4, 3], 2)
        num2_case6 = Number([2, 1], 1)
        quotient_case6 = Number([], 0)
        remainder_case6 = Number([], 0)
        large_real_division(16, num1_case6, num2_case6, quotient_case6, remainder_case6, 3)
        self.assertTrue(quotient_case6.num == [2, 0, 7] and remainder_case6.num == [0, 0, 0, 1, 9, 0])

    # def test_largediv4(self):
    #     num1_case1 = Number([1, 2, 5], 3)
    #     num2_case1 = Number([1, 2], 1)
    #     quotient_case1 = Number([], 0)
    #     remainder_case1 = Number([], 0)
    #     large_integer_division(10, num1_case1, num2_case1, quotient_case1, remainder_case1)
    #     self.assertTrue(quotient_case1.num == [1, 0] and remainder_case1.num == [0, 0, 0, 5])

    # def test_qr_decomposition(self):
    #     # Test Case 1: A simple 3x2 matrix
    #     A_case1 = np.array([[1, 2], [3, 4], [5, 6]])
    #     Q_case1, R_case1 = qr_decomposition(A_case1)
    #     self.assertTrue(np.allclose(np.dot(Q_case1, R_case1), A_case1))

    #     # Test Case 2: A square matrix
    #     A_case2 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    #     Q_case2, R_case2 = qr_decomposition(A_case2)
    #     self.assertTrue(np.allclose(np.dot(Q_case2, R_case2), A_case2))

    # def test_eig(self):
    #     # Test Case 1: A simple 2x2 matrix
    #     A_case1 = np.array([[2, 1], [1, 3]])
    #     eigvals_case1, eigvecs_case1 = eig(A_case1)
    #     self.assertTrue(np.allclose(np.dot(eigvecs_case1, np.dot(np.diag(eigvals_case1), eigvecs_case1.T)), A_case1))

    #     # Test Case 2: A symmetric matrix
    #     A_case2 = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    #     eigvals_case2, eigvecs_case2 = eig(A_case2)
    #     self.assertTrue(np.allclose(np.dot(eigvecs_case2, np.dot(np.diag(eigvals_case2), eigvecs_case2.T)), A_case2))

    def test_rank_of_matrix(self):
        # Test Case 1: Full-rank matrix
        mat_case1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result_case1 = rank_of_matrix(mat_case1)
        self.assertEqual(result_case1, 2)

        # Test Case 2: Rank-deficient matrix
        mat_case2 = [[1, 2, 3], [4, 5, 6], [0, 0, 0]]
        result_case2 = rank_of_matrix(mat_case2)
        self.assertEqual(result_case2, 2)

        # Test Case 3: Empty matrix
        mat_case3 = [[1, 2, 3], [4, 5, 6], [0, 1, 0]]
        result_case3 = rank_of_matrix(mat_case3)
        self.assertEqual(result_case3, 3)

    def test_SVD(self):
        # Test Case 1: A simple 2x3 matrix
        A_case1 = np.array([[0.89411765, 0.84117647, 0.77058824, 0.54411765, 0.91176471,
                      0.784375, 0.7625, 0.61785714, 0.69444444, 0.76666667,
                      0.6875, 0.58181818, 0.62222222, 0.44, 0.6,
                      0.48333333, 0.45, 0.41666667, 0.48333333, 0.5,
                      0.28571429, 0.47166667, 0.52166667, 0.45, 0.72142857],
                     [0.75543947, 0.63583333, 0.56670589, 0.52787547, 0.82429796,
                      0.86477368, 0.69059243, 0.64201142, 0.6202247, 0.63628521,
                      0.65243902, 0.45059658, 0.59061035, 0.54418605, 0.61658815,
                      0.23793103, 0.24285714, 0.15517241, 0.21785714, 0.12962963,
                      0.33333333, 0.50517084, 0.43846154, 0.34615385, 0.57777778],
                     [0.81939394, 0.67545455, 0.65515152, 0.65939394, 0.83606061,
                      0.803, 0.809, 0.61724138, 0.5875, 0.675,
                      0.66666667, 0.42884615, 0.66086957, 0.6, 0.55,
                      0.19047619, 0.2, 0.225, 0.195, 0.15789474,
                      0.31666667, 0.4568, 0.455, 0.5, 0.45714286],
                     [0.77526316, 0.69210526, 0.61526316, 0.72684211, 0.9,
                      0.67692308, 0.58571429, 0.54545455, 0.53333333, 0.52222222,
                      0.75555556, 0.30111111, 0.47777778, 0.56666667, 0.75555556,
                      0.14571429, 0.11714286, 0.04857143, 0.12142857, 0.03333333,
                      0.33333333, 0.38, 0.35, 0.45, 0.52857143]])
        U_case1, s_case1, sigma_case1, V_T_case1 = SVD(A_case1)
        self.assertTrue(np.allclose(np.dot(U_case1, np.dot(sigma_case1, V_T_case1)), A_case1) and np.allclose(s_case1, [5.67001081, 0.69305945, 0.31886819, 0.22939447]))

        # Test Case 2: A square matrix
        A_case2 = np.array([[0.2, -0.1, 0], [-0.1, 0.2, -0.1], [0, -0.1, 0.2]])
        U_case2, s_case2, sigma_case2, V_T_case2 = SVD(A_case2)
        self.assertTrue(np.allclose(np.dot(U_case2, np.dot(sigma_case2, V_T_case2)), A_case2) and np.allclose(s_case2, [0.34142136, 0.2, 0.05857864]))

        # Test Case 3: A rectangular matrix
        A_case3 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        U_case3, s_case3, sigma_case3, V_T_case3 = SVD(A_case3)
        self.assertTrue(np.allclose(np.dot(U_case3, np.dot(sigma_case3, V_T_case3)), A_case3) and np.allclose(s_case3, [0.95255181, 0.05143006]))

    def test_ReducedSVD(self):
        # Test Case 1: A simple 2x3 matrix with hyperparameter removal
        A_case1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        U_case1, s_case1, sigma_case1, V_T_case1 = ReducedSVD(A_case1, to_remove=1)
        self.assertTrue(np.allclose(np.dot(U_case1, np.dot(sigma_case1, V_T_case1)), [[0.15745455, 0.20801128, 0.258568], [0.37593612, 0.49664461, 0.6173531 ]])\
                        and np.allclose(s_case1, [0.9508032]))

        # Test Case 2: A square matrix with threshold removal
        A_case2 = np.array([[0.2, -0.1, 0], [-0.1, 0.2, -0.1], [0, -0.1, 0.2]])
        U_case2, s_case2, sigma_case2, V_T_case2 = ReducedSVD(A_case2, threshold=0.2)
        self.assertTrue(np.allclose(np.dot(U_case2, np.dot(sigma_case2, V_T_case2)), [[ 0.18535534, -0.12071068, -0.01464466],
 [-0.12071068,  0.17071068, -0.12071068],
 [-0.01464466 ,-0.12071068, 0.18535534]]) and np.allclose(s_case2,[0.34142136, 0.2]))

        # Test Case 3: A rectangular matrix with both removal methods
        A_case2 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        U_case2, s_case2, sigma_case2, V_T_case2 = ReducedSVD(A_case2, to_remove = 1, threshold=0.5)
        self.assertTrue(np.allclose(np.dot(U_case2, np.dot(sigma_case2, V_T_case2)), [[0.13566283, 0.17184622],
 [0.30971975, 0.39232681],
 [0.48377666, 0.61280741]]) and np.allclose(s_case2, [0.95255181]))