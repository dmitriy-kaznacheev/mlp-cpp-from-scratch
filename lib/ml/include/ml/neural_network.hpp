#pragma once

#include "matrix/matrix.hpp"
#include <functional>
#include <string>
#include <vector>

namespace ml {

struct LayerConfig {
  size_t input_size;
  size_t output_size;
  std::string activation;
};

struct TrainingMetrics {
  double loss = 0.0;
  double accuracy = 0.0;
  size_t epoch = 0;
};

class NeuralNetwork {
public:
  NeuralNetwork() = default;
  explicit NeuralNetwork(const std::vector<LayerConfig> &config);

  math::Matrix forward(const math::Matrix &input);
  void backward(const math::Matrix &input, const math::Matrix &target,
                double learning_rate);

  TrainingMetrics
  train(const std::vector<math::Matrix> &inputs,
        const std::vector<math::Matrix> &targets, size_t epochs,
        double learning_rate, size_t batch_size = 32,
        std::function<void(const TrainingMetrics &)> callback = nullptr);

  int predict(const math::Matrix &input);
  std::vector<double> predict_proba(const math::Matrix &input);
  double evaluate(const std::vector<math::Matrix> &inputs,
                  const std::vector<int> &labels);

  void save(const std::string &filepath) const;
  void load(const std::string &filepath);

  size_t num_layers() const { return weights_.size(); }
  size_t num_parameters() const;

private:
  std::vector<math::Matrix> weights_;
  std::vector<math::Matrix> biases_;
  std::vector<std::string> activations_;

  mutable std::vector<math::Matrix> layer_inputs_;
  mutable std::vector<math::Matrix> layer_outputs_;

  math::Matrix activate(const math::Matrix &x, const std::string &name) const;
  math::Matrix activate_derivative(const math::Matrix &x,
                                   const std::string &name) const;
  void initialize_weights(const std::vector<LayerConfig> &config);
  double cross_entropy_loss(const math::Matrix &output,
                            const math::Matrix &target) const;
};

} // namespace ml