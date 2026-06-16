#pragma once

#include <QImage>
#include <QPoint>
#include <QWidget>

class DrawingCanvas : public QWidget {
  Q_OBJECT

public:
  explicit DrawingCanvas(QWidget *parent = nullptr);

  QImage getImage() const;
  void clear();

protected:
  void paintEvent(QPaintEvent *event) override;
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void mouseReleaseEvent(QMouseEvent *event) override;

private:
  void drawLine(const QPoint &from, const QPoint &to);

private:
  QImage image_;
  QPoint last_point_;
  bool drawing_ = false;

  static const int CANVAS_SIZE = 280;
  static const int PEN_WIDTH = 20;
};