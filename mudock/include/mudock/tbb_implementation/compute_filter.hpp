#pragma once

#include <memory>
#include <mudock/batch.hpp>
#include <mudock/log.hpp>
#include <mudock/molecule.hpp>
#include <stdexcept>

namespace mudock {

  // The filter must be serial: stage is stateful and not thread-safe.
  template<typename stage_t>
  class compute_filter {
    std::shared_ptr<stage_t> stage;

  public:
    explicit compute_filter(std::shared_ptr<stage_t> s): stage(std::move(s)) {}

    std::unique_ptr<static_molecule> operator()(std::unique_ptr<static_molecule> ligand) const {
      if (!ligand)
        return nullptr;

      batch<static_molecule> b;
      b.batch_max_atoms    = ligand->num_atoms();
      b.batch_max_rotamers = static_cast<int>(ligand->num_rotamers());
      b.num_ligands        = 1;
      b.molecules[0]       = std::move(ligand);

      try {
        stage->prepare(b);
        (*stage)();
        stage->teardown(b);
      } catch (const std::runtime_error& e) {
        mudock::error("Unable to score ligand: ", e.what());
        return nullptr;
      }

      return std::move(b.molecules[0]);
    }
  };

} // namespace mudock
