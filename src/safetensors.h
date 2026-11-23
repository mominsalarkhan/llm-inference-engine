#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class DType { F32, F16, BF16, I64, I32, U8 };

const char* dtype_name(DType d);
size_t dtype_size(DType d);

// A view into the mapped file. No copy, no ownership.
struct Tensor {
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;

    int64_t numel() const;
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);
    ~SafeTensors();

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;

    bool has(const std::string& name) const;
    const Tensor& get(const std::string& name) const;
    const std::unordered_map<std::string, Tensor>& tensors() const { return tensors_; }

private:
    void parse(const uint8_t* base);

    void* map_ = nullptr;
    size_t size_ = 0;
    std::unordered_map<std::string, Tensor> tensors_;
};

// All weight files of one model folder: model.safetensors, or the shards listed in
// model.safetensors.index.json (bigger models ship split into several files).
class WeightFiles {
public:
    explicit WeightFiles(const std::string& model_dir);

    bool has(const std::string& name) const;
    const Tensor& get(const std::string& name) const;
    size_t file_count() const { return files_.size(); }

private:
    std::vector<std::unique_ptr<SafeTensors>> files_;
    std::unordered_map<std::string, const Tensor*> index_;
};

// Convert count elements starting at element `start` into float.
void to_f32(const Tensor& t, size_t start, size_t count, float* out);
std::vector<float> to_f32(const Tensor& t);
