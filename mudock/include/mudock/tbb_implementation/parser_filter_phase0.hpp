#pragma once

#include <atomic>
#include <memory>
#include <mudock/format/reader.hpp>
#include <mudock/format/supported_format.hpp>
#include <mudock/log.hpp>
#include <mudock/molecule.hpp>
#include <string>

namespace mudock {

  // Variant of parser_filter for the phase0 pipeline: instead of enqueuing into
  // a safe_queue, it returns one parsed molecule directly to the pipeline.
  // One token from stream_filter contains at most one molecule record, so the
  // 1-to-1 filter contract is satisfied. On parse failure returns nullptr.
  template<supported_format format>
  class parser_filter_phase0 {
    std::atomic<std::size_t>* skipped_ligands = nullptr;
    std::atomic<bool>* stop_requested         = nullptr;

  public:
    explicit parser_filter_phase0(std::atomic<std::size_t>* skipped = nullptr,
                                  std::atomic<bool>* stop           = nullptr)
        : skipped_ligands(skipped), stop_requested(stop) {}

    std::unique_ptr<static_molecule> operator()(std::string token) const {
      if (stop_requested != nullptr && stop_requested->load(std::memory_order_relaxed))
        return nullptr;

      // extract the first (and normally only) molecule record from the token
      type_of_format<format> splitter;
      std::string_view sv{token};
      const auto next     = splitter.next_molecule_start_index(sv);
      const auto mol_text = (next == std::string_view::npos) ? sv : sv.substr(0, next);

      if (mol_text.empty())
        return nullptr;

      try {
        return std::make_unique<static_molecule>(mudock::parser<format, static_molecule>(mol_text));
      } catch (const std::exception& e) {
        mudock::error("phase0 parse error: ", e.what());
        if (skipped_ligands != nullptr)
          skipped_ligands->fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      } catch (...) {
        mudock::error("phase0 parse error: unknown exception");
        if (skipped_ligands != nullptr)
          skipped_ligands->fetch_add(1, std::memory_order_relaxed);
        return nullptr;
      }
    }
  };

} // namespace mudock
