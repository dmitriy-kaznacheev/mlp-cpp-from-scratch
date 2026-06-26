#include "matrix/matrix.hpp"
#include "ml/mnist_loader.hpp"
#include "ml/neural_network.hpp"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ml;
using namespace ml::math;

void print_usage(const char *program) {
  std::cout << "Usage:\n"
            << "  Train:  " << program
            << " train <data_dir> <model_path> [epochs] [lr]\n"
            << "  Predict: " << program
            << " predict <model_path> <image_path>\n"
            << "\nExamples:\n"
            << "  " << program << " train ../data model.bin 20 0.001\n"
            << "  " << program << " predict model.bin digit.pgm\n";
}

Matrix load_pgm_image(const std::string &filepath) {
  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open image: " + filepath);
  }

  std::string format;
  int width, height, max_val;
  file >> format >> width >> height >> max_val;
  file.get(); // skip whitespace

  if (format != "P5") {
    throw std::runtime_error("Only PGM P5 format supported");
  }

  Matrix img(784, 1);
  for (int i = 0; i < 784; ++i) {
    unsigned char pixel;
    file.read(reinterpret_cast<char *>(&pixel), 1);

    // инвертирование: PGM (0=чёрный, 255=белый), MNIST (0=белый, 255=чёрный)
    double inverted = 255.0 - static_cast<double>(pixel);
    img(i, 0) = (inverted / 255.0 - 0.5) * 2.0;
  }
  return img;
}

void evaluate_model(const NeuralNetwork &network, const std::string &data_dir) {
  std::cout << "\nEvaluating on test dataset...\n";

  auto test_data = ml::MNISTLoader::load_dataset(
      data_dir + "/t10k-images-idx3-ubyte",
      data_dir + "/t10k-labels-idx1-ubyte", 10000);

  double test_accuracy = network.evaluate(test_data.images, test_data.labels);
  std::cout << "Test accuracy: " << test_accuracy * 100 << "%\n";

  auto cm = network.confusion_matrix(test_data.images, test_data.labels);

  std::cout << "\nConfusion Matrix:\n";
  std::cout << "    ";
  for (int j = 0; j < NeuralNetwork::NUM_CLASSES; ++j) {
    std::cout << std::setw(5) << j;
  }
  std::cout << "  | Total\n";

  for (int i = 0; i < NeuralNetwork::NUM_CLASSES; ++i) {
    std::cout << std::setw(2) << i << " |";
    int row_sum = 0;
    for (int j = 0; j < NeuralNetwork::NUM_CLASSES; ++j) {
      row_sum += cm[i][j];
      if (cm[i][j] > 0) {
        std::cout << std::setw(5) << cm[i][j];
      } else {
        std::cout << "     ";
      }
    }
    std::cout << "  | " << cm[i][i] << "/" << row_sum;
    std::cout << "\n";
  }
}

void train(const std::string &data_dir, const std::string &model_path,
           size_t epochs, double lr) {
  std::cout << "Loading MNIST dataset...\n";

  auto dataset =
      MNISTLoader::load_dataset(data_dir + "/train-images-idx3-ubyte",
                                data_dir + "/train-labels-idx1-ubyte", 60000);

  auto targets = MNISTLoader::one_hot_encode(dataset.labels);

  std::cout << "Loaded " << dataset.size() << " examples\n";

  std::vector<LayerConfig> config = {
      {784, 64, "relu"}, {64, 32, "relu"}, {32, 10, "softmax"}};

  NeuralNetwork network(config);
  std::cout << "Parameters: " << network.num_parameters() << "\n";
  std::cout << "Training " << epochs << " epochs, lr=" << lr << "...\n\n";

  network.train(
      dataset.images, targets, epochs, lr, 64, [](const TrainingMetrics &m) {
        std::cout << "Epoch " << std::setw(2) << m.epoch
                  << " | Loss: " << std::setw(10) << std::fixed
                  << std::setprecision(6) << m.loss
                  << " | Accuracy: " << std::setw(7) << std::fixed
                  << std::setprecision(2) << m.accuracy * 100 << "%\n";

        static std::ofstream log("../docs/training.log");
        log << m.epoch << " " << m.loss << " " << m.accuracy * 100 << "\n";
      });

  network.save(model_path);
  std::cout << "\nModel saved to " << model_path << "\n";

  evaluate_model(network, data_dir);
}

void predict(const std::string &model_path, const std::string &image_path) {
  NeuralNetwork network;
  network.load(model_path);

  Matrix image = load_pgm_image(image_path);
  int prediction = network.predict(image);
  auto probabilities = network.predict_proba(image);

  std::cout << "\nPredicted digit: " << prediction << "\n\n";
  std::cout << "Probabilities:\n";
  for (int i = 0; i < 10; ++i) {
    int bar = static_cast<int>(probabilities[i] * 50);
    std::cout << "  " << i << ": " << std::string(bar, '#')
              << std::string(50 - bar, ' ') << " " << probabilities[i] * 100
              << "%\n";
  }
}

int main(int argc, char *argv[]) {
  try {
    if (argc < 2) {
      print_usage(argv[0]);
      return 1;
    }

    std::string mode = argv[1];

    if (mode == "train") {
      if (argc < 4) {
        print_usage(argv[0]);
        return 1;
      }
      std::string data_dir = argv[2];
      std::string model_path = argv[3];
      size_t epochs = argc > 4 ? std::stoul(argv[4]) : 20;
      double lr = argc > 5 ? std::stod(argv[5]) : 0.001;
      train(data_dir, model_path, epochs, lr);

    } else if (mode == "predict") {
      if (argc < 4) {
        print_usage(argv[0]);
        return 1;
      }
      std::string model_path = argv[2];
      std::string image_path = argv[3];
      predict(model_path, image_path);

    } else {
      std::cerr << "Unknown mode: " << mode << "\n";
      print_usage(argv[0]);
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}