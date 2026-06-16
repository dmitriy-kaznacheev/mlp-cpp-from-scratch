#pragma once

#include "ml/neural_network.hpp"
#include "ui_mainwindow.h"
#include <QMainWindow>
#include <memory>

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);

private slots:
  void onPredictClicked();
  void onClearClicked();

private:
  void loadModel();
  void resetPrediction();
  void updatePrediction(int digit, const std::vector<double> &probabilities);
  bool isCanvasEmpty(const QImage &image) const;
  QImage centerImage(const QImage &image) const;

private:
  Ui::MainWindow ui_;
  std::unique_ptr<ml::NeuralNetwork> model_;
};