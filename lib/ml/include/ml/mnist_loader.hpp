#pragma once

#include "matrix/matrix.hpp"
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace ml {

class MNISTLoader {
public:
  struct Dataset {
    std::vector<math::Matrix> images;
    std::vector<int> labels;
    size_t size() const { return images.size(); }
  };

  static Dataset load_dataset(const std::string &images_path,
                              const std::string &labels_path,
                              size_t max_samples = 0);

  static std::vector<math::Matrix>
  one_hot_encode(const std::vector<int> &labels, size_t num_classes = 10);

  static std::pair<Dataset, Dataset> train_test_split(const Dataset &dataset,
                                                      double train_ratio = 0.8);

private:
  static uint32_t read_uint32_big_endian(std::ifstream &file);
};

} // namespace ml