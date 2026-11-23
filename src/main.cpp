#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model.h"
#include "safetensors.h"

#ifdef ENGINE_CUDA
#include "model_cuda.h"
#endif

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

// Pulls "greedy_ids": [...] for one prompt out of golden/meta.json.
std::vector<int> golden_greedy_ids(const std::string& meta_path, const std::string& prompt) {
    std::ifstream f(meta_path);
    if (!f) throw std::runtime_error("can't open " + meta_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    size_t at = text.find("\"" + prompt + "\": {");
    if (at == std::string::npos) throw std::runtime_error("prompt " + prompt + " not in meta.json");
    at = text.find("\"greedy_ids\"", at);
    at = text.find('[', at);
    size_t end = text.find(']', at);

    std::vector<int> ids;
    const char* p = text.c_str() + at + 1;
    const char* stop = text.c_str() + end;
    while (p < stop) {
        char* next = nullptr;
        long v = std::strtol(p, &next, 10);
        if (next == p) {
            ++p;
            continue;
        }
        ids.push_back(int(v));
        p = next;
    }
    return ids;
}

int argmax(const std::vector<float>& v) {
    return int(std::max_element(v.begin(), v.end()) - v.begin());
}

struct Diff {
    float max_abs = 0.0f;
    float rel = 0.0f; // max_abs relative to the largest reference value
};

Diff compare(const std::vector<float>& ours, const std::vector<float>& ref) {
    Diff d;
    float ref_max = 1e-6f;
    for (size_t i = 0; i < ref.size(); ++i) {
        d.max_abs = std::max(d.max_abs, std::fabs(ours[i] - ref[i]));
        ref_max = std::max(ref_max, std::fabs(ref[i]));
    }
    d.rel = d.max_abs / ref_max;
    return d;
}

// Full forward pass vs golden, checkpoint by checkpoint, then greedy generation.
// Works for both Model (CPU) and CudaModel.
// strict: every checkpoint within tol and identical greedy output (full-precision backends).
// Quantized backends drift by design, so for them we report the numbers and only require
// the next token to match.
template <typename M>
int run_check(M& model, const SafeTensors& golden, const std::vector<int>& ids, const std::vector<int>& want_gen,
              float tol, bool strict) {
    const Config& c = model.config();
    std::printf("model: dim %d, %d layers, %d heads (%d kv), head_dim %d, vocab %d\n",
                c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab);

    // collect every checkpoint for every prompt position
    std::map<std::string, std::vector<float>> got;
    Tap tap = [&](const std::string& name, const float* data, int n) {
        auto& v = got[name];
        v.insert(v.end(), data, data + n);
    };

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> last;
    for (size_t pos = 0; pos < ids.size(); ++pos) last = model.forward(ids[pos], int(pos), tap);
    double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    bool ok = true;
    auto check_one = [&](const std::string& name) -> Diff {
        Diff d = compare(got.at(name), to_f32(golden.get(name)));
        if (d.rel > tol && strict) ok = false;
        return d;
    };

    Diff e = check_one("embed");
    std::printf("\n%-8s rel %.2e\n", "embed", e.rel);
    for (int i = 0; i < c.n_layers && (ok || !strict); ++i) {
        std::string p = "layer" + std::to_string(i) + ".";
        const char* parts[] = {"attn_norm", "attn_out", "mlp_norm", "mlp_out", "out"};
        std::printf("layer %2d", i);
        for (const char* part : parts) {
            Diff d = check_one(p + part);
            std::printf("  %s %.1e%s", part, d.rel, d.rel > tol ? " <-- FAIL" : "");
        }
        std::printf("\n");
    }
    if (!ok && strict) {
        std::printf("\nFAIL: stopped at the first layer that drifted past %.0e\n", tol);
        return 1;
    }
    ok = true; // non-strict: layer drift is reported, not failed

    Diff fn = check_one("final_norm");
    Diff lg = check_one("logits");
    std::printf("%-8s rel %.2e\n%-8s rel %.2e (max abs %.2e)\n", "final", fn.rel, "logits", lg.rel, lg.max_abs);

    std::vector<float> ref_logits = to_f32(golden.get("logits"));
    int vocab = c.vocab;
    std::vector<float> ref_last(ref_logits.end() - vocab, ref_logits.end());
    int ours_next = argmax(last);
    int ref_next = argmax(ref_last);
    std::printf("next token: ours %d, hf %d  %s\n", ours_next, ref_next, ours_next == ref_next ? "match" : "MISMATCH");
    if (ours_next != ref_next) ok = false;

    // greedy generation, reusing the KV cache from the prompt
    std::vector<int> gen;
    int tok = ours_next;
    t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < want_gen.size(); ++i) {
        gen.push_back(tok);
        if (i + 1 == want_gen.size()) break;
        tok = argmax(model.forward(tok, int(ids.size() + i)));
    }
    double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    bool gen_ok = gen == want_gen;
    std::printf("\ngreedy ours:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\ngreedy hf:  ");
    for (int t : want_gen) std::printf(" %d", t);
    std::printf("\n%s\n", gen_ok ? "generation matches" : "generation differs");
    if (!gen_ok && strict) ok = false;

    std::printf("\nprefill %zu tokens in %.2fs, generated %zu tokens at %.2f tok/s\n",
                ids.size(), prefill_s, gen.size(), (gen.size() - 1) / std::max(gen_s, 1e-9));
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

bool is_quantized(const std::string& backend) {
    return backend.find("-q") != std::string::npos;
}

// Builds the model for a backend name and hands it to fn. Keeps every command backend-agnostic.
template <typename F>
int with_model(const std::string& backend, const std::string& model_dir, int max_seq, F&& fn) {
    if (backend == "cpu") {
        Model model(model_dir, max_seq);
        std::printf("backend: cpu\n");
        return fn(model);
    }
#ifdef ENGINE_CUDA
    Quant q;
    bool known = true;
    if (backend == "cuda") q = Quant::BF16;
    else if (backend == "cuda-q8") q = Quant::Q8;
    else if (backend == "cuda-q4") q = Quant::Q4;
    else known = false;
    if (known) {
        auto t0 = std::chrono::steady_clock::now();
        CudaModel model(model_dir, max_seq, q);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("backend: %s (%s), load %.1fs, weights %.2f GB on GPU\n", backend.c_str(),
                    model.device_name().c_str(), s, model.weight_bytes() / 1e9);
        return fn(model);
    }
#endif
    std::printf("unknown or unavailable backend: %s (try cpu, cuda, cuda-q8, cuda-q4)\n", backend.c_str());
    return 1;
}

int check(const std::string& model_dir, const std::string& golden_dir, const std::string& prompt,
          const std::string& backend) {
    SafeTensors golden(golden_dir + "/" + prompt + ".safetensors");
    std::vector<int> want_gen = golden_greedy_ids(golden_dir + "/meta.json", prompt);

    const Tensor& ids_t = golden.get("input_ids");
    std::vector<int> ids(size_t(ids_t.numel()));
    std::memcpy(ids.data(), ids_t.data, ids.size() * 4);
    int max_seq = int(ids.size() + want_gen.size()) + 1;

    bool quant = is_quantized(backend);
    return with_model(backend, model_dir, max_seq, [&](auto& model) {
        return run_check(model, golden, ids, want_gen, quant ? 0.1f : 1e-3f, !quant);
    });
}

// Plain greedy decode for n tokens, no taps, just speed.
int bench(const std::string& model_dir, const std::string& backend, int n) {
    return with_model(backend, model_dir, n + 2, [&](auto& model) {
        int tok = 1; // BOS
        model.forward(tok, 0); // warm up
        auto t0 = std::chrono::steady_clock::now();
        for (int pos = 1; pos <= n; ++pos) tok = argmax(model.forward(tok, pos));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("%d tokens in %.2fs: %.2f tok/s (%.1f ms/token)\n", n, s, n / s, 1000.0 * s / n);
        return 0;
    });
}

double meta_number(const std::string& meta_path, const std::string& key) {
    std::ifstream f(meta_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    size_t at = text.find("\"" + key + "\"");
    if (at == std::string::npos) return NAN;
    at = text.find(':', at);
    return std::strtod(text.c_str() + at + 1, nullptr);
}

// Perplexity over golden/ppl_tokens.bin: how surprised the model is by real text.
// exp(mean negative log-likelihood of each next token). Lower is better.
int ppl(const std::string& model_dir, const std::string& golden_dir, const std::string& backend) {
    std::ifstream f(golden_dir + "/ppl_tokens.bin", std::ios::binary);
    if (!f) throw std::runtime_error("no " + golden_dir + "/ppl_tokens.bin, rerun tools/golden.py");
    std::vector<int32_t> toks;
    int32_t t;
    while (f.read(reinterpret_cast<char*>(&t), 4)) toks.push_back(t);
    if (toks.size() < 2) throw std::runtime_error("ppl_tokens.bin is too short");

    double hf = meta_number(golden_dir + "/meta.json", "ppl_hf");

    return with_model(backend, model_dir, int(toks.size()) + 1, [&](auto& model) {
        auto t0 = std::chrono::steady_clock::now();
        double nll = 0.0;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            const std::vector<float>& logits = model.forward(toks[i], int(i));
            double mx = *std::max_element(logits.begin(), logits.end());
            double sum = 0.0;
            for (float v : logits) sum += std::exp(double(v) - mx);
            double logp = double(logits[toks[i + 1]]) - mx - std::log(sum);
            nll -= logp;
        }
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double ours = std::exp(nll / double(toks.size() - 1));
        std::printf("perplexity over %zu tokens: %.4f", toks.size(), ours);
        if (!std::isnan(hf)) std::printf("  (hf fp32: %.4f, diff %+.2f%%)", hf, 100.0 * (ours - hf) / hf);
        std::printf("\n%.1fs\n", s);
        return 0;
    });
}

void usage() {
    std::printf("usage:\n");
    std::printf("  engine inspect <file.safetensors>\n");
    std::printf("  engine check-embed <model.safetensors> <golden.safetensors>\n");
    std::printf("  engine check <model_dir> <golden_dir> <prompt> [backend]\n");
    std::printf("  engine bench <model_dir> [backend] [tokens]\n");
    std::printf("  engine ppl <model_dir> <golden_dir> [backend]\n");
    std::printf("backends: cpu, cuda, cuda-q8, cuda-q4\n");
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
        if (cmd == "check" && (argc == 5 || argc == 6))
            return check(argv[2], argv[3], argv[4], argc == 6 ? argv[5] : "cpu");
        if (cmd == "bench" && argc >= 3 && argc <= 5)
            return bench(argv[2], argc >= 4 ? argv[3] : "cpu", argc == 5 ? std::atoi(argv[4]) : 128);
        if (cmd == "ppl" && (argc == 4 || argc == 5)) return ppl(argv[2], argv[3], argc == 5 ? argv[4] : "cpu");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    usage();
    return 1;
}
