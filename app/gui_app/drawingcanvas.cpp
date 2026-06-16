#include "drawingcanvas.hpp"
#include <QMouseEvent>
#include <QPainter>

DrawingCanvas::DrawingCanvas(QWidget *parent) : QWidget(parent) {
  setFixedSize(CANVAS_SIZE, CANVAS_SIZE);
  image_ = QImage(CANVAS_SIZE, CANVAS_SIZE, QImage::Format_Grayscale8);
  image_.fill(Qt::white);
}

void DrawingCanvas::clear() {
  image_.fill(Qt::white);
  update();
}

QImage DrawingCanvas::getImage() const {
  return image_.scaled(28, 28, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
}

void DrawingCanvas::paintEvent(QPaintEvent *) {
  QPainter painter(this);
  painter.drawImage(0, 0, image_);
}

void DrawingCanvas::mousePressEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    last_point_ = event->pos();
    drawing_ = true;
  }
}

void DrawingCanvas::mouseMoveEvent(QMouseEvent *event) {
  if (drawing_) {
    drawLine(last_point_, event->pos());
    last_point_ = event->pos();
    update();
  }
}

void DrawingCanvas::mouseReleaseEvent(QMouseEvent *event) {
  if (event->button() == Qt::LeftButton) {
    drawing_ = false;
  }
}

void DrawingCanvas::drawLine(const QPoint &from, const QPoint &to) {
  QPainter painter(&image_);
  painter.setPen(
      QPen(Qt::black, PEN_WIDTH, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
  painter.setRenderHint(QPainter::Antialiasing);
  painter.drawLine(from, to);
}