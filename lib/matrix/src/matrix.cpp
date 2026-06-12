#include "matrix/matrix.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ml {
namespace math {

//=== Конструкторы ========================================

Matrix::Matrix() : rows_{0}, cols_{0} {}

Matrix::Matrix(size_t rows, size_t cols, double initial)
    : rows_{rows}, cols_{cols}, data_(rows * cols, initial) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init)
    : rows_{init.size()}, cols_{init.begin()->size()}, data_(rows_ * cols_) {
  auto it = data_.begin();
  for (const auto &row : init) {
    if (row.size() != cols_) {
      throw std::invalid_argument("All rows must have the same size");
    }
    it = std::copy(row.begin(), row.end(), it);
  }
}

Matrix::Matrix(const Matrix &other)
    : rows_{other.rows_}, cols_{other.cols_}, data_(other.data_) {}

Matrix::Matrix(Matrix &&other) noexcept
    : rows_{other.rows_}, cols_{other.cols_}, data_(std::move(other.data_)) {
  other.rows_ = 0;
  other.cols_ = 0;
}

//=== Фабричные методы ====================================

Matrix Matrix::zeros(size_t rows, size_t cols) {
  return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
  return Matrix(rows, cols, 1.0);
}

Matrix Matrix::random(size_t rows, size_t cols, double min, double max) {
  Matrix result(rows, cols);
  result.randomize(min, max);
  return result;
}

//=== Заполнение ==========================================

void Matrix::fill(double value) {
  std::fill(data_.begin(), data_.end(), value);
}

void Matrix::randomize(double min, double max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(min, max);
  std::generate(data_.begin(), data_.end(), [&]() { return dis(gen); });
}

//=== Операторы присваивания ==============================

Matrix &Matrix::operator=(const Matrix &other) {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
  }
  return *this;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
    other.rows_ = 0;
    other.cols_ = 0;
  }
  return *this;
}

//=== Составные операторы =================================

Matrix &Matrix::operator+=(const Matrix &other) {
  validate_dimensions(other, "addition");
  std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(),
                 [](double a, double b) { return a + b; });
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &other) {
  validate_dimensions(other, "subtraction");
  std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(),
                 [](double a, double b) { return a - b; });
  return *this;
}

Matrix &Matrix::operator*=(double scalar) {
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 [scalar](double a) { return a * scalar; });
  return *this;
}

Matrix &Matrix::operator/=(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 [scalar](double a) { return a / scalar; });
  return *this;
}

//=== Доступ к элементам ==================================

double &Matrix::operator()(size_t row, size_t col) {
  return data_[get_index(row, col)];
}

const double &Matrix::operator()(size_t row, size_t col) const {
  return data_[get_index(row, col)];
}

double &Matrix::at(size_t row, size_t col) {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of bounds: (" +
                            std::to_string(row) + "," + std::to_string(col) +
                            ") in (" + std::to_string(rows_) + "x" +
                            std::to_string(cols_) + ")");
  }
  return data_[get_index(row, col)];
}

const double &Matrix::at(size_t row, size_t col) const {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("Matrix index out of bounds: (" +
                            std::to_string(row) + "," + std::to_string(col) +
                            ") in (" + std::to_string(rows_) + "x" +
                            std::to_string(cols_) + ")");
  }
  return data_[get_index(row, col)];
}

//=== Транспонирование ====================================

Matrix Matrix::transpose() const {
  Matrix result(cols_, rows_);
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      result(j, i) = (*this)(i, j);
    }
  }
  return result;
}

//=== Получение строк и столбцов ==========================

Matrix Matrix::get_row(size_t row) const {
  if (row >= rows_) {
    throw std::out_of_range("Row index out of bounds: " + std::to_string(row));
  }
  Matrix result(1, cols_);
  std::copy(data_.begin() + row * cols_, data_.begin() + (row + 1) * cols_,
            result.begin());
  return result;
}

Matrix Matrix::get_col(size_t col) const {
  if (col >= cols_) {
    throw std::out_of_range("Column index out of bounds: " +
                            std::to_string(col));
  }
  Matrix result(rows_, 1);
  for (size_t i = 0; i < rows_; ++i) {
    result(i, 0) = (*this)(i, col);
  }
  return result;
}

//=== Информация ==========================================

size_t Matrix::rows() const { return rows_; }
size_t Matrix::cols() const { return cols_; }
size_t Matrix::size() const { return rows_ * cols_; }
bool Matrix::is_empty() const { return data_.empty(); }

//=== Статистика ==========================================

double Matrix::sum() const {
  return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Matrix::mean() const { return is_empty() ? 0.0 : sum() / size(); }

double Matrix::min() const {
  return is_empty() ? 0.0 : *std::min_element(data_.begin(), data_.end());
}

double Matrix::max() const {
  return is_empty() ? 0.0 : *std::max_element(data_.begin(), data_.end());
}

//=== Сериализация ========================================

void Matrix::save_binary(std::ofstream &file) const {
  file.write(reinterpret_cast<const char *>(&rows_), sizeof(rows_));
  file.write(reinterpret_cast<const char *>(&cols_), sizeof(cols_));
  file.write(reinterpret_cast<const char *>(data_.data()),
             size() * sizeof(double));
}

Matrix Matrix::load_binary(std::ifstream &file) {
  size_t rows, cols;
  file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

  Matrix result(rows, cols);
  file.read(reinterpret_cast<char *>(result.data_.data()),
            rows * cols * sizeof(double));

  return result;
}

//=== Приватные методы ====================================

size_t Matrix::get_index(size_t row, size_t col) const {
  return row * cols_ + col;
}

void Matrix::validate_dimensions(const Matrix &other,
                                 const std::string &operation) const {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    throw std::invalid_argument(
        "Matrix dimensions must match for " + operation + ": (" +
        std::to_string(rows_) + "x" + std::to_string(cols_) + ") vs (" +
        std::to_string(other.rows_) + "x" + std::to_string(other.cols_) + ")");
  }
}

//=== Свободные функции ===================================

//--- Арифметические ---

Matrix operator+(const Matrix &a, const Matrix &b) {
  Matrix result(a);
  result += b;
  return result;
}

Matrix operator-(const Matrix &a, const Matrix &b) {
  Matrix result(a);
  result -= b;
  return result;
}

Matrix operator*(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument(
        "Matrix multiplication dimension mismatch: (" +
        std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + ") * (" +
        std::to_string(b.rows()) + "x" + std::to_string(b.cols()) + ")");
  }
  Matrix result(a.rows(), b.cols(), 0.0);
  for (size_t i = 0; i < a.rows(); ++i) {
    for (size_t k = 0; k < a.cols(); ++k) {
      double temp = a(i, k);
      if (temp != 0.0) {
        for (size_t j = 0; j < b.cols(); ++j) {
          result(i, j) += temp * b(k, j);
        }
      }
    }
  }
  return result;
}

Matrix operator*(const Matrix &matrix, double scalar) {
  Matrix result(matrix);
  result *= scalar;
  return result;
}

Matrix operator*(double scalar, const Matrix &matrix) {
  return matrix * scalar;
}

Matrix operator/(const Matrix &matrix, double scalar) {
  Matrix result(matrix);
  result /= scalar;
  return result;
}

//--- Поэлементные операции ---

Matrix hadamard(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument(
        "Hadamard product dimension mismatch: (" + std::to_string(a.rows()) +
        "x" + std::to_string(a.cols()) + ") vs (" + std::to_string(b.rows()) +
        "x" + std::to_string(b.cols()) + ")");
  }
  Matrix result(a.rows(), a.cols());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                 std::multiplies<double>());
  return result;
}

Matrix element_wise_divide(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument(
        "Element-wise division dimension mismatch: (" +
        std::to_string(a.rows()) + "x" + std::to_string(a.cols()) + ") vs (" +
        std::to_string(b.rows()) + "x" + std::to_string(b.cols()) + ")");
  }
  Matrix result(a.rows(), a.cols());
  std::transform(a.begin(), a.end(), b.begin(), result.begin(),
                 [](double x, double y) {
                   if (y == 0.0)
                     throw std::invalid_argument(
                         "Division by zero in element-wise division");
                   return x / y;
                 });
  return result;
}

//--- Функции активации ---

Matrix sigmoid(const Matrix &m) {
  Matrix result(m.rows(), m.cols());
  std::transform(m.begin(), m.end(), result.begin(),
                 [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
  return result;
}

Matrix sigmoid_derivative(const Matrix &m) {
  Matrix result(m.rows(), m.cols());
  std::transform(m.begin(), m.end(), result.begin(), [](double x) {
    double s = 1.0 / (1.0 + std::exp(-x));
    return s * (1.0 - s);
  });
  return result;
}

Matrix relu(const Matrix &m) {
  Matrix result(m.rows(), m.cols());
  std::transform(m.begin(), m.end(), result.begin(),
                 [](double x) { return std::max(0.0, x); });
  return result;
}

Matrix relu_derivative(const Matrix &m) {
  Matrix result(m.rows(), m.cols());
  std::transform(m.begin(), m.end(), result.begin(),
                 [](double x) { return x > 0.0 ? 1.0 : 0.0; });
  return result;
}

Matrix softmax(const Matrix &m) {
  Matrix result = m;
  double max_val = result.max();

  double sum = 0.0;
  for (auto &val : result) {
    val = std::exp(val - max_val);
    sum += val;
  }

  for (auto &val : result) {
    val /= sum;
  }

  return result;
}

//--- Сравнение ---

bool operator==(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols())
    return false;
  return std::equal(a.begin(), a.end(), b.begin(),
                    [](double x, double y) { return std::abs(x - y) < 1e-10; });
}

bool operator!=(const Matrix &a, const Matrix &b) { return !(a == b); }

//--- Вывод ---

std::ostream &operator<<(std::ostream &os, const Matrix &matrix) {
  for (size_t i = 0; i < matrix.rows(); ++i) {
    os << "[ ";
    for (size_t j = 0; j < matrix.cols(); ++j) {
      os << matrix(i, j);
      if (j < matrix.cols() - 1)
        os << ", ";
    }
    os << " ]";
    if (i < matrix.rows() - 1)
      os << "\n";
  }
  return os;
}

} // namespace math
} // namespace ml