#include "mainwindow.hpp"
#include "drawingcanvas.hpp"
#include "matrix/matrix.hpp"
#include <QDebug>
#include <QMessageBox>
#include <QStatusBar>
#include <string_view>

static constexpr const char *MODEL_PATH = "../data/model.bin";
static constexpr int MNIST_SIZE = 28;
static constexpr int MNIST_PIXELS = MNIST_SIZE * MNIST_SIZE; // 784

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
  ui_.setupUi(this);
  layout()->setSizeConstraint(QLayout::SetFixedSize);

  connect(ui_.predictButton, &QPushButton::clicked, this,
          &MainWindow::onPredictClicked);
  connect(ui_.clearButton, &QPushButton::clicked, this,
          &MainWindow::onClearClicked);

  loadModel();
  resetPrediction();
}

void MainWindow::loadModel() {
  try {
    model_ = std::make_unique<ml::NeuralNetwork>();
    model_->load(MODEL_PATH);
    statusBar()->showMessage(QString("Модель загружена: %1").arg(MODEL_PATH));
  } catch (const std::exception &e) {
    QMessageBox::warning(this, "Ошибка",
                         QString("Модель не загружена: %1").arg(e.what()));
    ui_.predictButton->setEnabled(false);
  }
}

void MainWindow::resetPrediction() {
  ui_.predictionLabel->setText("Нарисуйте цифру");
  ui_.confidenceBar->hide();
  ui_.probabilitiesLabel->clear();
}

void MainWindow::onPredictClicked() {
  if (!model_) {
    return;
  }

  try {
    QImage image = ui_.canvas->getImage();
    image = centerImage(image);

    if (isCanvasEmpty(image)) {
      resetPrediction();
      return;
    }

    ml::math::Matrix input(MNIST_PIXELS, 1);
    for (int y = 0; y < MNIST_SIZE; ++y) {
      for (int x = 0; x < MNIST_SIZE; ++x) {
        int pixel = qGray(image.pixel(x, y));
        double inverted = 255.0 - static_cast<double>(pixel);
        input(y * MNIST_SIZE + x, 0) = (inverted / 255.0 - 0.5) * 2.0;
      }
    }

    auto prediction = model_->predict(input);
    auto probabilities = model_->predict_proba(input);
    updatePrediction(prediction, probabilities);
  } catch (const std::exception &e) {
    QMessageBox::critical(this, "Error", e.what());
  }
}

void MainWindow::onClearClicked() {
  ui_.canvas->clear();
  resetPrediction();
}

void MainWindow::updatePrediction(int digit,
                                  const std::vector<double> &probabilities) {
  ui_.predictionLabel->setText(QString("Результат: %1").arg(digit));

  {
    auto confidence = static_cast<int>(probabilities[digit] * 100);
    ui_.confidenceBar->setValue(confidence);
    ui_.confidenceBar->show();

    QString color = (confidence > 90)   ? "mediumseagreen"
                    : (confidence > 70) ? "steelblue"
                                        : "indianred";

    ui_.confidenceBar->setStyleSheet(
        QString("QProgressBar::chunk { background-color: %1; }").arg(color));
  }

  QString text = "Вероятности по классам:<br>";
  {
    static constexpr int BAR_WIDTH = 20;
    for (int i = 0; i < 10; ++i) {
      auto bar_len = static_cast<int>(probabilities[i] * BAR_WIDTH);
      auto empty_len = BAR_WIDTH - bar_len;
      QString color = (i == digit) ? "mediumseagreen" : "darkgray";

      text += QString("<pre style='margin:0; color:%1;'>%2: %3%4 %5%</pre>")
                  .arg(color)
                  .arg(i)
                  .arg(QString(bar_len, QChar(0x2588)))
                  .arg(QString(empty_len, ' '))
                  .arg(probabilities[i] * 100, 5, 'f', 1);
    }
  }
  ui_.probabilitiesLabel->setText(text);
}

bool MainWindow::isCanvasEmpty(const QImage &image) const {
  static constexpr int EMPTY_THRESHOLD = 200;
  for (int y = 0; y < MNIST_SIZE; ++y) {
    for (int x = 0; x < MNIST_SIZE; ++x) {
      if (qGray(image.pixel(x, y)) < EMPTY_THRESHOLD) {
        return false;
      }
    }
  }
  return true;
}

QImage MainWindow::centerImage(const QImage &image) const {
  auto min_x = MNIST_SIZE, max_x = 0, min_y = MNIST_SIZE, max_y = 0;
  auto has_content = false;
  static constexpr int DARK_THRESHOLD = 128;
  for (int y = 0; y < MNIST_SIZE; ++y) {
    for (int x = 0; x < MNIST_SIZE; ++x) {
      if (qGray(image.pixel(x, y)) < DARK_THRESHOLD) {
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        has_content = true;
      }
    }
  }

  if (!has_content) {
    return image;
  }

  auto content_w = max_x - min_x + 1;
  auto content_h = max_y - min_y + 1;
  auto offset_x = (MNIST_SIZE - content_w) / 2 - min_x;
  auto offset_y = (MNIST_SIZE - content_h) / 2 - min_y;

  QImage centered(MNIST_SIZE, MNIST_SIZE, QImage::Format_Grayscale8);
  centered.fill(Qt::white);

  auto in_bounds = [](int x, int y) {
    return x >= 0 && x < MNIST_SIZE && y >= 0 && y < MNIST_SIZE;
  };

  for (int y = 0; y < MNIST_SIZE; ++y) {
    for (int x = 0; x < MNIST_SIZE; ++x) {
      auto src_x = x - offset_x;
      auto src_y = y - offset_y;
      if (in_bounds(src_x, src_y)) {
        centered.setPixel(x, y, image.pixel(src_x, src_y));
      }
    }
  }

  return centered;
}