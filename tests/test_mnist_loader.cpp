#include "ml/mnist_loader.hpp"
#include <fstream>
#include <gtest/gtest.h>

using namespace ml;
using namespace ml::math;

class MNISTLoaderTest : public ::testing::Test {
protected:
  void SetUp() override { create_test_files(); }

  void TearDown() override {
    std::remove("test_images.idx3-ubyte");
    std::remove("test_labels.idx1-ubyte");
  }

  void create_test_files() {
    // 10 изображений 28x28
    {
      std::ofstream f("test_images.idx3-ubyte", std::ios::binary);
      write_uint32_big_endian(f, 2051); // magic
      write_uint32_big_endian(f, 10);   // count
      write_uint32_big_endian(f, 28);   // rows
      write_uint32_big_endian(f, 28);   // cols

      for (size_t i = 0; i < 10 * 28 * 28; ++i) {
        unsigned char pixel = static_cast<unsigned char>(i % 256);
        f.write(reinterpret_cast<char *>(&pixel), 1);
      }
    }

    // 10 меток
    {
      std::ofstream f("test_labels.idx1-ubyte", std::ios::binary);
      write_uint32_big_endian(f, 2049); // magic
      write_uint32_big_endian(f, 10);   // count

      for (unsigned char i = 0; i < 10; ++i) {
        f.write(reinterpret_cast<char *>(&i), 1);
      }
    }
  }

  void write_uint32_big_endian(std::ofstream &file, uint32_t value) {
    uint32_t big_endian = ((value & 0xFF) << 24) | ((value & 0xFF00) << 8) |
                          ((value & 0xFF0000) >> 8) |
                          ((value & 0xFF000000) >> 24);
    file.write(reinterpret_cast<char *>(&big_endian), sizeof(big_endian));
  }
};

TEST_F(MNISTLoaderTest, load_dataset) {
  auto dataset = MNISTLoader::load_dataset("test_images.idx3-ubyte",
                                           "test_labels.idx1-ubyte");

  EXPECT_EQ(dataset.images.size(), 10);
  EXPECT_EQ(dataset.labels.size(), 10);

  for (const auto &img : dataset.images) {
    EXPECT_EQ(img.rows(), 784);
    EXPECT_EQ(img.cols(), 1);
  }
}

TEST_F(MNISTLoaderTest, load_dataset_max_samples) {
  auto dataset = MNISTLoader::load_dataset("test_images.idx3-ubyte",
                                           "test_labels.idx1-ubyte", 5);

  EXPECT_EQ(dataset.size(), 5);
}

TEST_F(MNISTLoaderTest, onehot_encode) {
  auto labels = std::vector<int>{0, 1, 2};
  auto encoded = MNISTLoader::one_hot_encode(labels, 10);

  EXPECT_EQ(encoded.size(), 3);
  EXPECT_DOUBLE_EQ(encoded[0](0, 0), 1.0);
  EXPECT_DOUBLE_EQ(encoded[0](1, 0), 0.0);
  EXPECT_DOUBLE_EQ(encoded[1](0, 0), 0.0);
  EXPECT_DOUBLE_EQ(encoded[1](1, 0), 1.0);
}

TEST_F(MNISTLoaderTest, train_test_split) {
  auto dataset = MNISTLoader::load_dataset("test_images.idx3-ubyte",
                                           "test_labels.idx1-ubyte");

  auto [train, test] = MNISTLoader::train_test_split(dataset, 0.7);

  EXPECT_EQ(train.size() + test.size(), dataset.size());
  EXPECT_NEAR(static_cast<double>(train.size()) / dataset.size(), 0.7, 0.1);
}