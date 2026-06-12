#pragma once

#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ml {
namespace math {

class Matrix {
public:
  // Конструкторы
  Matrix();
  Matrix(size_t rows, size_t cols, double initial = 0.0);
  Matrix(std::initializer_list<std::initializer_list<double>> init);
  Matrix(const Matrix &other);
  Matrix(Matrix &&other) noexcept;

  // Фабричные методы
  static Matrix zeros(size_t rows, size_t cols);
  static Matrix ones(size_t rows, size_t cols);
  static Matrix random(size_t rows, size_t cols, double min = 0.0,
                       double max = 1.0);

  // Заполнение
  void fill(double value);
  void randomize(double min = 0.0, double max = 1.0);

  // Операторы присваивания
  Matrix &operator=(const Matrix &other);
  Matrix &operator=(Matrix &&other) noexcept;

  // Составные операторы
  Matrix &operator+=(const Matrix &other);
  Matrix &operator-=(const Matrix &other);
  Matrix &operator*=(double scalar);
  Matrix &operator/=(double scalar);

  // Доступ к элементам
  double &operator()(size_t row, size_t col);
  const double &operator()(size_t row, size_t col) const;
  double &at(size_t row, size_t col);
  const double &at(size_t row, size_t col) const;

  // Транспонирование
  Matrix transpose() const;

  // Получение строк и столбцов
  Matrix get_row(size_t row) const;
  Matrix get_col(size_t col) const;

  // Информация
  size_t rows() const;
  size_t cols() const;
  size_t size() const;
  bool is_empty() const;

  // Статистика
  double sum() const;
  double mean() const;
  double min() const;
  double max() const;

  // Сериализация
  void save_binary(std::ofstream &file) const;
  static Matrix load_binary(std::ifstream &file);

  // Итераторы
  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }

private:
  size_t get_index(size_t row, size_t col) const;
  void validate_dimensions(const Matrix &other,
                           const std::string &operation) const;

private:
  size_t rows_{0};
  size_t cols_{0};
  std::vector<double> data_;
};

//=== Свободные функции ===================================

// Арифметические
Matrix operator+(const Matrix &a, const Matrix &b);
Matrix operator-(const Matrix &a, const Matrix &b);
Matrix operator*(const Matrix &a, const Matrix &b);
Matrix operator*(const Matrix &matrix, double scalar);
Matrix operator*(double scalar, const Matrix &matrix);
Matrix operator/(const Matrix &matrix, double scalar);

// Поэлементные
Matrix hadamard(const Matrix &a, const Matrix &b);
Matrix element_wise_divide(const Matrix &a, const Matrix &b);

// Активации
Matrix sigmoid(const Matrix &m);
Matrix sigmoid_derivative(const Matrix &m);
Matrix relu(const Matrix &m);
Matrix relu_derivative(const Matrix &m);
Matrix softmax(const Matrix &m);

// Сравнение
bool operator==(const Matrix &a, const Matrix &b);
bool operator!=(const Matrix &a, const Matrix &b);

// Вывод
std::ostream &operator<<(std::ostream &os, const Matrix &matrix);

} // namespace math
} // namespace ml