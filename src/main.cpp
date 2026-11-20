#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "safetensors.h"

namespace {

std::string shape_str(const std::vector<int64_t>& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(shape[i]);
    }
    return s + "]";
}

int inspect(const std::string& path) {
    SafeTensors st(path);

    std::vector<std::string> names;
    for (const auto& kv : st.tensors()) names.push_back(kv.first);
    std::sort(names.begin(), names.end());

    int64_t params = 0;
    size_t bytes = 0;
    for (const auto& name : names) {
        const Tensor& t = st.get(name);
        std::printf("%-50s %-5s %s\n", name.c_str(), dtype_name(t.dtype), shape_str(t.shape).c_str());
        params += t.numel();
        bytes += t.nbytes;
    }
    std::printf("\n%zu tensors, %.3fB params, %.2f GB\n", names.size(), params / 1e9, bytes / 1e9);
    return 0;
}

// First real check: our embedding lookup vs the Hugging Face one.
int check_embed(const std::string& model_path, const std::string& golden_path) {
    SafeTensors model(model_path);
    SafeTensors golden(golden_path);

    const Tensor& table = model.get("model.embed_tokens.weight");
    const Tensor& ids = golden.get("input_ids");
    std::vector<float> ref = to_f32(golden.get("embed"));

    size_t dim = size_t(table.shape[1]);
    size_t n = size_t(ids.numel());
    if (ref.size() != n * dim) {
        std::printf("shape mismatch: golden embed has %zu values, expected %zu\n", ref.size(), n * dim);
        return 1;
    }

    std::vector<float> ours(n * dim);
    for (size_t i = 0; i < n; ++i) {
        int32_t tok;
        std::memcpy(&tok, ids.data + i * 4, 4);
        to_f32(table, size_t(tok) * dim, dim, &ours[i * dim]);
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < ours.size(); ++i) max_diff = std::max(max_diff, std::fabs(ours[i] - ref[i]));

    bool ok = max_diff < 1e-6f;
    std::printf("embed: %zu tokens x %zu dims, max diff %g  %s\n", n, dim, max_diff, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

void usage() {
    std::printf("usage:\n");
    std::printf("  engine inspect <file.safetensors>\n");
    std::printf("  engine check-embed <model.safetensors> <golden.safetensors>\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    std::string cmd = argv[1];
    try {
        if (cmd == "inspect" && argc == 3) return inspect(argv[2]);
        if (cmd == "check-embed" && argc == 4) return check_embed(argv[2], argv[3]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    usage();
    return 1;
}
