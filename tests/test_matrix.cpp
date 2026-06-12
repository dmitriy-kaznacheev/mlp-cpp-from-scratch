#include "matrix/matrix.hpp"
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>

using namespace ml::math;
using namespace std::string_literals;

class MatrixTest : public ::testing::Test {
protected:
  void SetUp() override {
    m2x2_ = Matrix{{1.0, 2.0}, {3.0, 4.0}};
    m2x3_ = Matrix{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    m3x2_ = Matrix{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  }

  Matrix m2x2_;
  Matrix m2x3_;
  Matrix m3x2_;
};

//=== Конструкторы ========================================

TEST_F(MatrixTest, default_constructor) {
  Matrix m;
  EXPECT_EQ(m.rows(), 0);
  EXPECT_EQ(m.cols(), 0);
  EXPECT_TRUE(m.is_empty());
}

TEST_F(MatrixTest, size_constructor) {
  auto m = Matrix(3, 4, 1.5);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 4);
  EXPECT_EQ(m.size(), 12);
}

TEST_F(MatrixTest, initializer_list_constructor) {
  EXPECT_EQ(m2x2_.rows(), 2);
  EXPECT_EQ(m2x2_.cols(), 2);
  EXPECT_DOUBLE_EQ(m2x2_(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m2x2_(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(m2x2_(1, 0), 3.0);
  EXPECT_DOUBLE_EQ(m2x2_(1, 1), 4.0);
}

TEST_F(MatrixTest, copy_constructor) {
  auto m = Matrix(m2x2_);
  EXPECT_TRUE(m == m2x2_);
}

TEST_F(MatrixTest, move_constructor) {
  auto m1 = Matrix{{1.0, 2.0}, {3.0, 4.0}};
  auto m2(std::move(m1));
  EXPECT_EQ(m2.rows(), 2);
  EXPECT_EQ(m2.cols(), 2);
  EXPECT_TRUE(m1.is_empty());
}

//=== Фабричные методы ====================================

TEST_F(MatrixTest, zeros) {
  auto m = Matrix::zeros(3, 3);
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_TRUE(
      std::all_of(m.begin(), m.end(), [](double v) { return v == 0.0; }));
}

TEST_F(MatrixTest, ones) {
  auto m = Matrix::ones(2, 4);
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 4);
  EXPECT_TRUE(
      std::all_of(m.begin(), m.end(), [](double v) { return v == 1.0; }));
}

TEST_F(MatrixTest, random) {
  auto m = Matrix::random(10, 10, -1.0, 1.0);
  auto min_val = m.min();
  auto max_val = m.max();
  EXPECT_GE(min_val, -1.0);
  EXPECT_LE(max_val, 1.0);
}

//=== Арифметические операции =============================

TEST_F(MatrixTest, addition) {
  auto m = m2x2_ + m2x2_;
  EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 6.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
}

TEST_F(MatrixTest, subtraction) {
  auto m = m2x2_ - m2x2_;
  EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 0.0);
}

TEST_F(MatrixTest, matrix_multiplication) {
  auto m = m2x3_ * m3x2_;
  EXPECT_EQ(m.rows(), 2);
  EXPECT_EQ(m.cols(), 2);
  EXPECT_DOUBLE_EQ(m(0, 0), 58.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 64.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 139.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 154.0);
}

TEST_F(MatrixTest, scalar_multiplication) {
  auto m = m2x2_ * 2.0;
  EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 6.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
}

TEST_F(MatrixTest, scalar_multiplication_left) {
  auto m = 3.0 * m2x2_;
  EXPECT_DOUBLE_EQ(m(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 12.0);
}

TEST_F(MatrixTest, scalar_division) {
  auto m = m2x2_ / 2.0;
  EXPECT_DOUBLE_EQ(m(0, 0), 0.5);
  EXPECT_DOUBLE_EQ(m(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 1.5);
  EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
}

//=== Составные операторы =================================

TEST_F(MatrixTest, addition_assignment) {
  auto m = m2x2_;
  m += m2x2_;
  EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 8.0);
}

TEST_F(MatrixTest, subtraction_assignment) {
  auto m = m2x2_;
  m -= m2x2_;
  EXPECT_DOUBLE_EQ(m(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 0.0);
}

TEST_F(MatrixTest, scalar_multiplication_assignment) {
  auto m = m2x2_;
  m *= 3.0;
  EXPECT_DOUBLE_EQ(m(0, 0), 3.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 12.0);
}

TEST_F(MatrixTest, scalar_division_assignment) {
  auto m = m2x2_;
  m /= 2.0;
  EXPECT_DOUBLE_EQ(m(0, 0), 0.5);
  EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
}

//=== Поэлементные операции ===============================

TEST_F(MatrixTest, hadamard) {
  auto m = hadamard(m2x2_, m2x2_);
  EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 9.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 16.0);
}

TEST_F(MatrixTest, element_Wise_divide) {
  auto a = Matrix{{4.0, 6.0}, {8.0, 10.0}};
  auto b = Matrix{{2.0, 3.0}, {4.0, 5.0}};
  auto m = element_wise_divide(a, b);
  EXPECT_DOUBLE_EQ(m(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 2.0);
  EXPECT_DOUBLE_EQ(m(1, 0), 2.0);
  EXPECT_DOUBLE_EQ(m(1, 1), 2.0);
}

//=== Транспонирование ====================================

TEST_F(MatrixTest, transpose) {
  auto m = m2x3_.transpose();
  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 2);
  EXPECT_DOUBLE_EQ(m(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(m(0, 1), 4.0);
}

TEST_F(MatrixTest, double_transpose) {
  auto m = m2x3_.transpose().transpose();
  EXPECT_TRUE(m == m2x3_);
}

//=== Функции активации ===================================

TEST_F(MatrixTest, sigmoid) {
  auto m1 = Matrix(1, 1, 0.0);
  auto m2 = sigmoid(m1);
  EXPECT_NEAR(m2(0, 0), 0.5, 1e-10);
}

TEST_F(MatrixTest, sigmoid_range) {
  auto m1 = Matrix::random(10, 10, -10.0, 10.0);
  auto m2 = sigmoid(m1);
  for (size_t i = 0; i < m2.rows(); ++i) {
    for (size_t j = 0; j < m2.cols(); ++j) {
      EXPECT_GT(m2(i, j), 0.0);
      EXPECT_LT(m2(i, j), 1.0);
    }
  }
}

TEST_F(MatrixTest, sigmoid_derivative) {
  auto m1 = Matrix(1, 1, 0.0);
  auto m2 = sigmoid_derivative(m1);
  EXPECT_NEAR(m2(0, 0), 0.25, 1e-10);
}

TEST_F(MatrixTest, relu) {
  auto m1 = Matrix{{-1.0, 0.0, 1.0}, {-2.0, 2.0, 3.0}};
  auto m2 = relu(m1);
  EXPECT_DOUBLE_EQ(m2(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(m2(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(m2(0, 2), 1.0);
  EXPECT_DOUBLE_EQ(m2(1, 0), 0.0);
  EXPECT_DOUBLE_EQ(m2(1, 1), 2.0);
  EXPECT_DOUBLE_EQ(m2(1, 2), 3.0);
}

TEST_F(MatrixTest, relu_derivative) {
  auto m1 = Matrix{{-1.0, 0.0, 1.0}};
  auto m2 = relu_derivative(m1);
  EXPECT_DOUBLE_EQ(m2(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(m2(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(m2(0, 2), 1.0);
}

TEST_F(MatrixTest, softmax) {
  auto m1 = Matrix{{1.0, 2.0, 3.0}};
  auto m2 = softmax(m1);

  auto sum = 0.0;
  for (size_t j = 0; j < m2.cols(); ++j) {
    EXPECT_GT(m2(0, j), 0.0);
    EXPECT_LT(m2(0, j), 1.0);
    sum += m2(0, j);
  }
  EXPECT_NEAR(sum, 1.0, 1e-10);
}

//=== Cтатистики ==========================================

TEST_F(MatrixTest, sum) { EXPECT_DOUBLE_EQ(m2x2_.sum(), 10.0); }

TEST_F(MatrixTest, mean) { EXPECT_DOUBLE_EQ(m2x2_.mean(), 2.5); }

TEST_F(MatrixTest, min) { EXPECT_DOUBLE_EQ(m2x2_.min(), 1.0); }

TEST_F(MatrixTest, max) { EXPECT_DOUBLE_EQ(m2x2_.max(), 4.0); }

//=== Получение строк и столбцов ==========================

TEST_F(MatrixTest, get_row) {
  auto row = m2x2_.get_row(0);
  EXPECT_EQ(row.rows(), 1);
  EXPECT_EQ(row.cols(), 2);
  EXPECT_DOUBLE_EQ(row(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(row(0, 1), 2.0);
}

TEST_F(MatrixTest, get_col) {
  auto col = m2x2_.get_col(1);
  EXPECT_EQ(col.rows(), 2);
  EXPECT_EQ(col.cols(), 1);
  EXPECT_DOUBLE_EQ(col(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(col(1, 0), 4.0);
}

//=== Заполнение ==========================================

TEST_F(MatrixTest, fill) {
  auto m = Matrix(3, 3);
  m.fill(5.0);

  EXPECT_EQ(m.rows(), 3);
  EXPECT_EQ(m.cols(), 3);
  EXPECT_TRUE(
      std::all_of(m.begin(), m.end(), [](double v) { return v == 5.0; }));
}

//=== Сравнение ===========================================

TEST_F(MatrixTest, equal) {
  auto m1 = Matrix{{1.0, 2.0}, {3.0, 4.0}};
  auto m2 = Matrix{{1.0, 2.0}, {3.0, 4.0}};
  EXPECT_TRUE(m1 == m2);
}

TEST_F(MatrixTest, not_equal) {
  auto m1 = Matrix{{1.0, 2.0}, {3.0, 4.0}};
  auto m2 = Matrix{{1.0, 2.0}, {3.0, 5.0}};
  EXPECT_TRUE(m1 != m2);
}

//=== Вывод ===============================================

TEST_F(MatrixTest, output_stream) {
  auto m = Matrix{{1.0, 2.0}, {3.0, 4.0}};
  std::ostringstream oss;
  oss << m;
  EXPECT_FALSE(oss.str().empty());
}

//=== Исключения ==========================================

TEST_F(MatrixTest, simension_mismatch) {
  auto m3x3 = Matrix(3, 3);
  auto m4x4 = Matrix(4, 4);

  EXPECT_THROW(m2x2_ + m3x3, std::invalid_argument);
  EXPECT_THROW(m2x2_ - m3x3, std::invalid_argument);
  EXPECT_THROW(m2x2_ * m3x3, std::invalid_argument);
  EXPECT_THROW(m2x3_ * m4x4, std::invalid_argument);
  EXPECT_THROW(hadamard(m2x2_, m3x3), std::invalid_argument);
}

TEST_F(MatrixTest, out_of_bounds) {
  EXPECT_THROW(m2x2_.at(5, 5), std::out_of_range);
  EXPECT_THROW(m2x2_.get_row(5), std::out_of_range);
  EXPECT_THROW(m2x2_.get_col(5), std::out_of_range);
}

TEST_F(MatrixTest, division_by_zero) {
  EXPECT_THROW(m2x2_ / 0.0, std::invalid_argument);
  EXPECT_THROW(m2x2_ /= 0.0, std::invalid_argument);
}

TEST_F(MatrixTest, element_wise_divide_by_zero) {
  auto m1 = Matrix{{4.0, 6.0}};
  auto m2 = Matrix{{2.0, 0.0}};
  EXPECT_THROW(element_wise_divide(m1, m2), std::invalid_argument);
}

//=== Сериализация ========================================

TEST_F(MatrixTest, serialization) {
  const auto file_name = "test_matrix.bin"s;

  {
    std::ofstream ofs(file_name, std::ios::binary);
    m2x2_.save_binary(ofs);
  }

  Matrix loaded;
  {
    std::ifstream ifs(file_name, std::ios::binary);
    loaded = Matrix::load_binary(ifs);
  }

  EXPECT_TRUE(loaded == m2x2_);
  std::remove(file_name.c_str());
}