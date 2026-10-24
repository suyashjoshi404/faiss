#include "hnswlib/hnswlib.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string dataset = "sift";
    fs::path data_file, query_file, groundtruth_file, output_file, performance_file,
            layer_averages_file, index_file;
    int M = 32, ef_construction = 128, ef_search = 64, k = 10, warmup = 500;
    bool check_data = false, build_only = false, search_only = false, force = false;
};

static uint32_t read_u32(std::istream& in, const std::string& context) {
    uint32_t value = 0;
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("Truncated " + context);
    return value;
}

struct VectorFile {
    fs::path path;
    bool fvecs;
    uint32_t count, dimension;
    std::ifstream input;

    VectorFile(const fs::path& filename, bool is_fvecs) : path(filename), fvecs(is_fvecs), input(filename, std::ios::binary) {
        if (!input) throw std::runtime_error("Cannot open " + path.string());
        const uintmax_t bytes = fs::file_size(path);
        if (fvecs) {
            dimension = read_u32(input, path.string());
            if (!dimension || bytes % (4ULL * (dimension + 1)))
                throw std::runtime_error("Invalid fvecs size: " + path.string());
            count = bytes / (4ULL * (dimension + 1));
            input.seekg(0);
        } else {
            const uint32_t element_bytes = read_u32(input, path.string());
            count = read_u32(input, path.string());
            dimension = read_u32(input, path.string());
            if (element_bytes != sizeof(float) || !count || !dimension ||
                bytes != 12ULL + 4ULL * count * dimension)
                throw std::runtime_error("Invalid data_new header or size: " + path.string());
        }
    }

    void next(std::vector<float>& vector) {
        if (fvecs && read_u32(input, path.string()) != dimension)
            throw std::runtime_error("Inconsistent fvecs dimension: " + path.string());
        vector.resize(dimension);
        if (!input.read(reinterpret_cast<char*>(vector.data()), sizeof(float) * dimension))
            throw std::runtime_error("Truncated vectors: " + path.string());
    }
};

struct TruthFile {
    fs::path path;
    uint32_t count, width;
    std::ifstream input;
    explicit TruthFile(const fs::path& filename) : path(filename), input(filename, std::ios::binary) {
        if (!input) throw std::runtime_error("Cannot open " + path.string());
        const auto bytes = fs::file_size(path);
        width = read_u32(input, path.string());
        if (!width || bytes % (4ULL * (width + 1)))
            throw std::runtime_error("Invalid ivecs size: " + path.string());
        count = bytes / (4ULL * (width + 1));
        input.seekg(0);
    }
    std::vector<uint32_t> next() {
        if (read_u32(input, path.string()) != width)
            throw std::runtime_error("Inconsistent ivecs width: " + path.string());
        std::vector<uint32_t> row(width);
        if (!input.read(reinterpret_cast<char*>(row.data()), 4ULL * width))
            throw std::runtime_error("Truncated ground truth: " + path.string());
        return row;
    }
};

static int integer_arg(const char* value, const char* name, int minimum) {
    size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != std::string(value).size() || parsed < minimum)
        throw std::runtime_error(std::string("Invalid ") + name + ": " + value);
    return parsed;
}

static Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--check-data") { opts.check_data = true; continue; }
        if (key == "--build-only") { opts.build_only = true; continue; }
        if (key == "--search-only") { opts.search_only = true; continue; }
        if (key == "--force") { opts.force = true; continue; }
        if (key == "--help") {
            std::cout << "Usage: shg_profile --dataset sift|enron [--M 32] [--ef-construction 128] "
                         "[--ef-search 64] [--k 10] [--warmup 500] [--data-file PATH] "
                         "[--query-file PATH] [--groundtruth-file PATH] [--output PATH] "
                         "[--performance-output PATH] [--layer-averages-output PATH] [--index-file PATH] "
                         "[--build-only|--search-only] [--force] [--check-data]\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--dataset") opts.dataset = value;
        else if (key == "--M") opts.M = integer_arg(argv[i], "M", 2);
        else if (key == "--ef-construction") opts.ef_construction = integer_arg(argv[i], "ef-construction", 1);
        else if (key == "--ef-search") opts.ef_search = integer_arg(argv[i], "ef-search", 1);
        else if (key == "--k") opts.k = integer_arg(argv[i], "k", 1);
        else if (key == "--warmup") opts.warmup = integer_arg(argv[i], "warmup", 0);
        else if (key == "--data-file") opts.data_file = value;
        else if (key == "--query-file") opts.query_file = value;
        else if (key == "--groundtruth-file") opts.groundtruth_file = value;
        else if (key == "--output") opts.output_file = value;
        else if (key == "--performance-output") opts.performance_file = value;
        else if (key == "--layer-averages-output") opts.layer_averages_file = value;
        else if (key == "--index-file") opts.index_file = value;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (opts.dataset != "sift" && opts.dataset != "enron")
        throw std::runtime_error("--dataset must be sift or enron");
    if (opts.build_only && opts.search_only)
        throw std::runtime_error("Choose either --build-only or --search-only");
    if (opts.ef_search < opts.k)
        throw std::runtime_error("ef-search must be at least k");
    const fs::path root = "experiments/hnsw-profiling";
    const fs::path data_dir = root / "data" / opts.dataset;
    if (opts.data_file.empty()) opts.data_file = data_dir / (opts.dataset == "sift" ? "sift_base.fvecs" : "enron.data_new");
    if (opts.query_file.empty()) opts.query_file = data_dir / (opts.dataset + "_query.fvecs");
    if (opts.groundtruth_file.empty()) opts.groundtruth_file = data_dir / (opts.dataset + "_groundtruth.ivecs");
    if (opts.output_file.empty()) opts.output_file = root / "results" /
        (opts.dataset + "_shg_M" + std::to_string(opts.M) + "_ef" + std::to_string(opts.ef_search) + "_layer_stats.csv");
    if (opts.performance_file.empty()) opts.performance_file = root / "observations" /
        (opts.dataset + "_shg_performance.csv");
    if (opts.layer_averages_file.empty()) opts.layer_averages_file = root / "observations" /
        (opts.dataset + "_shg_layer_averages.csv");
    if (opts.index_file.empty()) opts.index_file = root / "indexes" / opts.dataset /
        (opts.dataset + "_shg_M" + std::to_string(opts.M) + ".index");
    return opts;
}

static void write_build_performance(const Options& opts, uint32_t count, uint32_t dimension,
                                    double build_seconds, double shortcut_seconds) {
    if (!opts.performance_file.parent_path().empty()) fs::create_directories(opts.performance_file.parent_path());
    const bool header = !fs::exists(opts.performance_file) || fs::file_size(opts.performance_file) == 0;
    std::ofstream out(opts.performance_file, std::ios::app);
    if (!out) throw std::runtime_error("Cannot write " + opts.performance_file.string());
    if (header) out << "dataset,algorithm,phase,count,dimension,M,ef_construction,ef_search,k,threads,"
                       "seconds,average_latency_us,queries_per_second,recall_at_k,shortcut_build_seconds\n";
    out << std::fixed << std::setprecision(8) << opts.dataset << ",shg,build," << count << ','
        << dimension << ',' << opts.M << ',' << opts.ef_construction << ",,,,"
        << build_seconds << ",,,," << shortcut_seconds << '\n';
}

static double recall_at_k(std::priority_queue<std::pair<float, hnswlib::labeltype>> labels,
                          const std::vector<uint32_t>& truth, int k) {
    std::unordered_set<uint32_t> expected(truth.begin(), truth.begin() + k);
    int hits = 0;
    while (!labels.empty()) {
        hits += expected.count(static_cast<uint32_t>(labels.top().second));
        labels.pop();
    }
    return static_cast<double>(hits) / k;
}

static uint64_t sum_stat(const hnswlib::SHGSearchStats& stats,
                         uint64_t hnswlib::SHGSearchStats::Layer::* field) {
    uint64_t total = 0;
    for (const auto& layer : stats.layers) total += layer.*field;
    return total;
}

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        std::unique_ptr<VectorFile> base;
        if (!opts.search_only || opts.check_data)
            base = std::make_unique<VectorFile>(opts.data_file, opts.dataset == "sift" || opts.data_file.extension() == ".fvecs");
        std::unique_ptr<VectorFile> query_file;
        std::unique_ptr<TruthFile> truth_file;
        if (!opts.build_only) {
            query_file = std::make_unique<VectorFile>(opts.query_file, true);
            truth_file = std::make_unique<TruthFile>(opts.groundtruth_file);
            if ((base && base->dimension != query_file->dimension) ||
                query_file->count != truth_file->count || opts.k > static_cast<int>(truth_file->width))
                throw std::runtime_error("Base/query dimensions, query/truth counts, or k/truth width disagree");
        }
        if (opts.check_data) {
            std::cout << "Dataset: " << opts.dataset << ", base: " << base->count
                      << ", queries: " << query_file->count << ", dimension: " << base->dimension
                      << ", ground truth width: " << truth_file->width << '\n';
            return 0;
        }
        if (opts.build_only && !opts.force &&
            (fs::exists(opts.index_file) || fs::exists(opts.index_file.string() + ".shg")))
            throw std::runtime_error("Index already exists; use --force to rebuild: " + opts.index_file.string());

        const auto build_start = Clock::now();
        hnswlib::L2Space space(base ? base->dimension : query_file->dimension);
        std::unique_ptr<hnswlib::HEDS<float>> holder;
        if (opts.search_only) {
            holder = std::make_unique<hnswlib::HEDS<float>>(&space, opts.index_file.string(), false,
                       static_cast<size_t>(query_file->count) + 1);
            if (holder->cur_element_count + query_file->count > holder->max_elements_)
                holder->resizeIndex(holder->cur_element_count + query_file->count);
            holder->loadSearchState(opts.index_file.string() + ".shg", query_file->dimension);
        } else {
            holder = std::make_unique<hnswlib::HEDS<float>>(&space, base->dimension,
                       static_cast<size_t>(base->count) + (query_file ? query_file->count : 0),
                       opts.M, opts.ef_construction);
        }
        auto& index = *holder;
        Performance performance;
        std::vector<float> vector;
        if (!opts.search_only) {
            for (uint32_t id = 0; id < base->count; ++id) {
                base->next(vector);
                index.addDataPoint(vector.data(), id, 1, performance);
            }
        }
        const double build_seconds = opts.search_only ? 0 :
            std::chrono::duration<double>(Clock::now() - build_start).count();
        const uint32_t base_count = opts.search_only ? index.cur_element_count.load() : base->count;
        double shortcut_seconds = 0;
        if (!opts.search_only) {
            const auto shortcut_start = Clock::now();
            int mutable_base_count = base_count;
            index.buildShortcuts(mutable_base_count);
            shortcut_seconds = std::chrono::duration<double>(Clock::now() - shortcut_start).count();
            if (opts.build_only) {
                if (!opts.index_file.parent_path().empty()) fs::create_directories(opts.index_file.parent_path());
                index.saveIndex(opts.index_file.string());
                index.saveSearchState(opts.index_file.string() + ".shg");
                write_build_performance(opts, base_count, base->dimension, build_seconds, shortcut_seconds);
                std::cout << "Index saved to " << opts.index_file << " and " << opts.index_file << ".shg\n"
                          << "Build time: " << build_seconds << " s\n"
                          << "Shortcut build time: " << shortcut_seconds << " s\n";
                return 0;
            }
            write_build_performance(opts, base_count, base->dimension, build_seconds, shortcut_seconds);
        }
        index.setEf(opts.ef_search);

        // SHG needs query representations in its storage. flag=-1 stores them
        // without adding graph links; do this outside measured search.
        std::vector<std::vector<uint32_t>> truth;
        truth.reserve(query_file->count);
        for (uint32_t id = 0; id < query_file->count; ++id) {
            query_file->next(vector);
            index.addDataPoint(vector.data(), static_cast<size_t>(base_count) + id, -1, performance);
            truth.push_back(truth_file->next());
        }
        const int max_level = index.maxlevel_;
        const int layers = max_level + 1;
        std::vector<uint32_t> nodes(layers, 0);
        for (uint32_t id = 0; id < base_count; ++id)
            for (int layer = 0; layer <= index.element_levels_[id]; ++layer) nodes[layer]++;
        index.resultsProcessing.assign(static_cast<size_t>(base_count + query_file->count) * layers, -1);
        const auto search = [&](uint32_t id, hnswlib::SHGSearchStats* stats) {
            Query query(static_cast<int>(base_count + id), opts.k);
            return index.searchKnnShortcuts(query, nullptr, stats);
        };
        for (uint32_t id = 0; id < std::min<uint32_t>(opts.warmup, query_file->count); ++id)
            search(id, nullptr);
        // Only the post-warmup query pass is included in search_seconds.
        // Per-query stats, recall calculation, and CSV output happen later.
        const auto search_start = Clock::now();
        for (uint32_t id = 0; id < query_file->count; ++id) search(id, nullptr);
        const double search_seconds = std::chrono::duration<double>(Clock::now() - search_start).count();

        if (!opts.output_file.parent_path().empty()) fs::create_directories(opts.output_file.parent_path());
        std::ofstream out(opts.output_file);
        if (!out) throw std::runtime_error("Cannot write " + opts.output_file.string());
        out << "dataset,algorithm,query_id,layer,nodes_on_layer,max_level,num_layers,M,ef_search,k,"
               "hops,distance_computations,full_distance_computations,compressed_distance_computations,"
               "pruned_neighbors,levels_skipped,total_query_hops,total_query_distance_computations,recall_at_k\n";
        out << std::fixed << std::setprecision(8);
        uint64_t total_hops = 0, total_distances = 0;
        double total_recall = 0;
        std::vector<uint64_t> layer_hops(layers, 0), layer_distances(layers, 0);
        for (uint32_t id = 0; id < query_file->count; ++id) {
            hnswlib::SHGSearchStats stats(layers);
            const auto result = search(id, &stats);
            const double recall = recall_at_k(result, truth[id], opts.k);
            const auto query_hops = sum_stat(stats, &hnswlib::SHGSearchStats::Layer::hops);
            const auto query_distances = sum_stat(stats, &hnswlib::SHGSearchStats::Layer::distance_computations);
            total_hops += query_hops;
            total_distances += query_distances;
            total_recall += recall;
            for (int layer = 0; layer < layers; ++layer) {
                const auto& s = stats.layers[layer];
                layer_hops[layer] += s.hops;
                layer_distances[layer] += s.distance_computations;
                out << opts.dataset << ",shg," << id << ',' << layer << ',' << nodes[layer] << ','
                    << max_level << ',' << layers << ',' << index.M_ << ',' << opts.ef_search << ',' << opts.k << ','
                    << s.hops << ',' << s.distance_computations << ',' << s.full_distance_computations << ','
                    << s.compressed_distance_computations << ',' << s.pruned_neighbors << ',' << s.levels_skipped
                    << ',' << query_hops << ',' << query_distances << ',' << recall << '\n';
            }
        }
        const double nq = query_file->count;
        const double average_recall = total_recall / nq;
        const double queries_per_second = nq / search_seconds;
        if (!opts.performance_file.parent_path().empty())
            fs::create_directories(opts.performance_file.parent_path());
        const bool performance_header = !fs::exists(opts.performance_file) ||
                                        fs::file_size(opts.performance_file) == 0;
        std::ofstream performance_out(opts.performance_file, std::ios::app);
        if (!performance_out) throw std::runtime_error("Cannot write " + opts.performance_file.string());
        if (performance_header)
            performance_out << "dataset,algorithm,phase,count,dimension,M,ef_construction,ef_search,k,threads,"
                               "seconds,average_latency_us,queries_per_second,recall_at_k,shortcut_build_seconds\n";
        performance_out << std::fixed << std::setprecision(8)
                        << opts.dataset << ",shg,search," << query_file->count << ',' << query_file->dimension << ','
                        << index.M_ << ',' << index.ef_construction_ << ',' << opts.ef_search << ',' << opts.k
                        << ",1," << search_seconds << ',' << search_seconds / nq * 1e6 << ','
                        << queries_per_second << ',' << average_recall << ',' << (opts.search_only ? "" : std::to_string(shortcut_seconds)) << '\n';
        if (!opts.layer_averages_file.parent_path().empty())
            fs::create_directories(opts.layer_averages_file.parent_path());
        const bool averages_header = !fs::exists(opts.layer_averages_file) ||
                                     fs::file_size(opts.layer_averages_file) == 0;
        std::ofstream averages_out(opts.layer_averages_file, std::ios::app);
        if (!averages_out) throw std::runtime_error("Cannot write " + opts.layer_averages_file.string());
        if (averages_header)
            averages_out << "dataset,algorithm,M,ef_search,k,num_queries,layer,nodes_on_layer,"
                            "average_hops,average_distance_computations,recall_at_k,queries_per_second\n";
        averages_out << std::fixed << std::setprecision(8);
        for (int layer = 0; layer < layers; ++layer)
            averages_out << opts.dataset << ",shg," << index.M_ << ',' << opts.ef_search << ',' << opts.k << ','
                         << query_file->count << ',' << layer << ',' << nodes[layer] << ','
                         << static_cast<double>(layer_hops[layer]) / query_file->count << ','
                         << static_cast<double>(layer_distances[layer]) / query_file->count << ','
                         << average_recall << ',' << queries_per_second << '\n';
        std::cout << std::fixed << std::setprecision(6);
        if (!opts.search_only)
            std::cout << "Build time: " << build_seconds << " s\n"
                      << "Shortcut build time: " << shortcut_seconds << " s\n";
        std::cout << "Batch search time: " << search_seconds << " s\n"
                  << "Performance CSV: " << opts.performance_file << '\n'
                  << "Average latency: " << search_seconds / nq * 1e6 << " us/query\n"
                  << "Queries/sec: " << nq / search_seconds << '\n'
                  << "Average total hops: " << total_hops / nq << '\n'
                  << "Average distance comps: " << total_distances / nq << '\n'
                  << "Recall@" << opts.k << ": " << total_recall / nq << '\n';
        for (int layer = 0; layer < layers; ++layer)
            std::cout << "  layer " << layer << ": hops=" << layer_hops[layer] / nq
                      << ", distance_computations=" << layer_distances[layer] / nq << '\n';
        std::cout << "Per-layer averages CSV: " << opts.layer_averages_file << '\n';
        std::cout << "Per-query/per-layer CSV: " << opts.output_file << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shg_profile: " << error.what() << '\n';
        return 1;
    }
}
