#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <istream>
#include <limits>
#include <memory>
#include <mudock/compute/manager.hpp>
#include <mudock/compute/scratchpad.hpp>
#include <mudock/format/supported_format.hpp>
#include <mudock/implementations.hpp>
#include <mudock/knobs.hpp>
#include <mudock/log.hpp>
#include <mudock/molecule.hpp>
#include <mudock/tbb_implementation/compute_filter.hpp>
#include <mudock/tbb_implementation/parser_filter_phase0.hpp>
#include <mudock/tbb_implementation/runtime_services.hpp>
#include <mudock/tbb_implementation/stream_filter.hpp>
#include <mudock/tbb_implementation/writer_filter.hpp>
#include <mudock/utils.hpp>
#include <oneapi/tbb/parallel_pipeline.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mudock {

  // Constructs a single stage for the given device configuration and invokes
  // callback(stage).
  // Mirrors the dispatch logic of manager but returns a stage instead of spawning a worker thread.
  template<typename pipeline_t, typename callback_t>
  void make_single_stage(const std::string& configuration,
                         const knobs& knobs,
                         pipeline_t& pipe,
                         callback_t&& callback) {
    const auto parts  = parse_worker_configuration(configuration);
    const auto dev_t  = get_device_type(parts[1]);
    const auto impl_t = get_impl_type(parts[0]);
    const int id      = parse_ids(parts[2])[0];

    if (dev_t == device_type::CPU) {
      constexpr_for<0, num_cpu_kernel_type(), 1>([&](const auto kernel_idx) {
        constexpr auto kt = cpu_kernel_type[kernel_idx];
        if (kt == impl_t) {
          using k_t        = typename kernel_type_traits<kt>::type;
          using stage_t    = decltype(pipe.template get_pipeline<k_t>(knobs,
                                                                   id,
                                                                   device_type::CPU,
                                                                   std::shared_ptr<scratchpad<k_t>>{}));
          auto dev_scratch = std::make_shared<scratchpad<k_t>>(knobs, 0, device_type::CPU);
          // make_shared: requires a movable / copiable type.
          // new + shared_ptr: in-place construction.
          auto stage_ptr = std::shared_ptr<stage_t>(
              new stage_t(pipe.template get_pipeline<k_t>(knobs, id, device_type::CPU, dev_scratch)));
          callback(stage_ptr);
        }
      });
    } else {
      constexpr_for<0, num_gpu_kernel_type(), 1>([&](const auto kernel_idx) {
        constexpr auto kt = gpu_kernel_type[kernel_idx];
        if (kt == impl_t) {
          using k_t        = typename kernel_type_traits<kt>::type;
          using stage_t    = decltype(pipe.template get_pipeline<k_t>(knobs,
                                                                   id,
                                                                   device_type::GPU,
                                                                   std::shared_ptr<scratchpad<k_t>>{}));
          auto dev_scratch = std::make_shared<scratchpad<k_t>>(knobs, id, device_type::GPU);
          auto stage_ptr   = std::shared_ptr<stage_t>(
              new stage_t(pipe.template get_pipeline<k_t>(knobs, id, device_type::GPU, dev_scratch)));
          callback(stage_ptr);
        }
      });
    }
  }

  // Intentional oversimplifications:
  //   - single device: only the first id for configurations[0] is used
  //   - compute filter is serial: one ligand processed at a time
  //   - no clustering: stage setup cost is paid per ligand
  template<supported_format format, typename pipeline_t>
  void run_tbb_pipeline_phase0(std::istream& in,
                               const std::vector<std::string>& configurations,
                               const knobs& knobs,
                               pipeline_t& pipe,
                               std::size_t end                      = std::numeric_limits<std::size_t>::max(),
                               std::optional<double> time_limit_sec = std::nullopt) {
    if (configurations.empty())
      throw std::runtime_error("run_tbb_pipeline_phase0: no device configuration provided");

    std::atomic<bool> stop_requested{false};
    std::atomic<std::size_t> skipped_ligands{0};

    detail::deadline_timer timer;
    if (time_limit_sec && *time_limit_sec > 0.0) {
      info("Time limit enabled: ", *time_limit_sec, " s");
      timer.start(time_limit_sec, [&]() {
        stop_requested.store(true, std::memory_order_relaxed);
        info("Time limit reached.");
      });
    }

    make_single_stage(configurations.front(), knobs, pipe, [&](auto stage_ptr) {
      using stage_t = typename decltype(stage_ptr)::element_type;
      std::string buf;
      buf.reserve(1 << 20);
      std::size_t written_lines = 0;

      const auto tbb_stream_filter = oneapi::tbb::make_filter<void, std::string>(
          oneapi::tbb::filter_mode::serial_in_order,
          stream_filter<format>(in, knobs.max_bytes_per_token, end, &stop_requested));
      const auto tbb_parser_filter = oneapi::tbb::make_filter<std::string, std::unique_ptr<static_molecule>>(
          oneapi::tbb::filter_mode::parallel,
          parser_filter_phase0<format>(&skipped_ligands, &stop_requested));
      const auto tbb_compute_filter =
          oneapi::tbb::make_filter<std::unique_ptr<static_molecule>, std::unique_ptr<static_molecule>>(
              oneapi::tbb::filter_mode::serial_out_of_order,
              compute_filter<stage_t>(stage_ptr));
      const auto tbb_writer_filter = oneapi::tbb::make_filter<std::unique_ptr<static_molecule>, void>(
          oneapi::tbb::filter_mode::serial_in_order,
          writer_filter(buf, written_lines));

      oneapi::tbb::parallel_pipeline(knobs.max_tbb_tokens,
                                     tbb_stream_filter & tbb_parser_filter & tbb_compute_filter &
                                         tbb_writer_filter);

      if (!buf.empty())
        std::cout << buf;
    });

    timer.cancel();
    timer.join();

    if (const auto skipped = skipped_ligands.load(std::memory_order_relaxed); skipped > 0)
      error("Skipped ", skipped, " ligand(s) due to parse errors.");

    info("Output drained");
  }

} // namespace mudock
