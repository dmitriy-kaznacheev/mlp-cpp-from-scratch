#include "ml/neural_network.hpp"
#include <fstream>
#include <gtest/gtest.h>

using namespace ml;
using namespace ml::math;

class NeuralNetworkTest : public ::testing::Test {
protected:
  void SetUp() override {
    config_ = {{10, 5, "relu"}, {5, 3, "softmax"}};
    network_ = std::make_unique<NeuralNetwork>(config_);

    input_ = Matrix::random(10, 1, 0.0, 1.0);
    target_ = Matrix::zeros(3, 1);
    target_(0, 0) = 1.0;
  }

  std::vector<LayerConfig> config_;
  std::unique_ptr<NeuralNetwork> network_;
  Matrix input_;
  Matrix target_;
};

TEST_F(NeuralNetworkTest, constructor) {
  EXPECT_EQ(network_->num_layers(), 2);
  EXPECT_GT(network_->num_parameters(), 0);
}

TEST_F(NeuralNetworkTest, constructor_empty_config) {
  EXPECT_THROW(NeuralNetwork(std::vector<LayerConfig>{}), std::invalid_argument);
}

TEST_F(NeuralNetworkTest, forward) {
  auto output = network_->forward(input_);
  EXPECT_EQ(output.rows(), 3);
  EXPECT_EQ(output.cols(), 1);

  auto sum = 0.0;
  for (size_t i = 0; i < output.rows(); ++i) {
    EXPECT_GE(output(i, 0), 0.0);
    EXPECT_LE(output(i, 0), 1.0);
    sum += output(i, 0);
  }
  EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(NeuralNetworkTest, predict) {
  auto prediction = network_->predict(input_);
  EXPECT_GE(prediction, 0);
  EXPECT_LE(prediction, 2);
}

TEST_F(NeuralNetworkTest, predict_proba) {
  auto probs = network_->predict_proba(input_);
  EXPECT_EQ(probs.size(), 3);

  auto sum = 0.0;
  for (auto p : probs) {
    sum += p;
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
  }
  EXPECT_NEAR(sum, 1.0, 1e-6);
}

TEST_F(NeuralNetworkTest, backward) {
  network_->forward(input_);
  EXPECT_NO_THROW(network_->backward(input_, target_, 0.01));
}

TEST_F(NeuralNetworkTest, serialization) {
  network_->save("test_model.bin");

  NeuralNetwork loaded(config_);
  loaded.load("test_model.bin");

  auto pred1 = network_->predict(input_);
  auto pred2 = loaded.predict(input_);
  EXPECT_EQ(pred1, pred2);

  std::remove("test_model.bin");
}

TEST_F(NeuralNetworkTest, training_loop) {
  std::vector<Matrix> inputs = {input_};
  std::vector<Matrix> targets = {target_};

  auto metrics = network_->train(inputs, targets, 5, 0.01, 1);
  EXPECT_GE(metrics.accuracy, 0.0);
  EXPECT_LE(metrics.accuracy, 1.0);
}

TEST_F(NeuralNetworkTest, different_architectures) {
  auto configs = std::vector<std::vector<LayerConfig>>{
      {{10, 10, "relu"}, {10, 3, "softmax"}},
      {{10, 8, "sigmoid"}, {8, 3, "softmax"}},
      {{10, 15, "relu"}, {15, 10, "relu"}, {10, 3, "softmax"}}};

  for (const auto &cfg : configs) {
    auto net = NeuralNetwork(cfg);
    auto out = net.forward(input_);
    EXPECT_EQ(out.rows(), 3);
  }
}

TEST_F(NeuralNetworkTest, Evaluate) {
  std::vector<Matrix> inputs = {input_};
  std::vector<int> labels = {0};

  auto accuracy = network_->evaluate(inputs, labels);
  EXPECT_GE(accuracy, 0.0);
  EXPECT_LE(accuracy, 1.0);
}