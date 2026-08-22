# Changelog

Behaviour changes that a caller could observe, announced rather than slipped
in. Newest first. Entries name the work package of the design document that
decided them.

## Unreleased

### Changed

- **Z-Image `res_2s` output changes (deliberate).** `RES2sScheduler` is an
  exponential integrator in the data prediction `x₀ = x − σ·v`; `ZImagePipeline`
  and `ZImageControlPipeline` were feeding it the flow velocity, which is
  dimensionally wrong. Every `ZImageScheduler` now declares a
  `modelOutputConvention` (`.velocity` for all existing schedulers,
  `.dataPrediction` for `res_2s`) and the pipelines convert once per model
  evaluation through `modelInput(velocity:sample:sigma:)`. A Z-Image render with
  `scheduler: res_2s` therefore differs from earlier builds — measured in
  `ZImageRES2sCorrectionTests` (pre-fix feed misses the exact-field x₀ by >1.0
  relative; post-fix reconstructs it to ≤1e-5). The default euler/flow path and
  every other sampler are byte-identical. FDD-krea2-raw-recipe D2 / §3.2, WP-E2,
  AC-10/11/74.
