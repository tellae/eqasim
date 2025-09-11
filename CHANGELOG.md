# Changelog

## [1.3.0](https://github.com/tellae/eqasim/compare/v1.2.0...v1.3.0) (2025-05-07)


### Features

* add comments for bpe enriched ([f121451](https://github.com/tellae/eqasim/commit/f121451d842d1f39bb78ee9b8c04d4440c88784a))
* add config for Rennes ([d916c50](https://github.com/tellae/eqasim/commit/d916c5097aea90fc371d833f0778249bc2f61f55))
* add documentation for Rennes ([6440c20](https://github.com/tellae/eqasim/commit/6440c20172f70f73b20fe2ca7903d00de50978e5))
* artificially add trips to reach 4 millions ([f0ddd39](https://github.com/tellae/eqasim/commit/f0ddd39ff7a894f91f673f188f841852fa0e7ca5))
* bpe enriched with open data from RM ([dad7d4a](https://github.com/tellae/eqasim/commit/dad7d4aee511f762c74f012dcbd5a489387c9b3a))
* comment code to save files ([db6a963](https://github.com/tellae/eqasim/commit/db6a96358f42b3dbbf71e0ac4b9273d72d2fd4e8))
* scripts to use emc² ([895981a](https://github.com/tellae/eqasim/commit/895981a36a4ed767b451806b71d67196479f6585))


### Bug Fixes

* Arbitrary order of week days in merged GTFS ([#131](https://github.com/tellae/eqasim/issues/131)) ([f963e3b](https://github.com/tellae/eqasim/commit/f963e3b2eeb06fbabf15ff872610c9df0d3b5535))
* Behaviour of shutil.which ([#128](https://github.com/tellae/eqasim/issues/128)) ([2879603](https://github.com/tellae/eqasim/commit/2879603d10dc8e5b178b86e89fe7ce42dfd37d01))
* Properly treat non-movers in EDGT 44 ([#133](https://github.com/tellae/eqasim/issues/133)) ([09ed87a](https://github.com/tellae/eqasim/commit/09ed87ae47703d519fc90ef7080f775358412a73))
* Remaining bug when loading BPE 2019 ([#132](https://github.com/tellae/eqasim/issues/132)) ([3457a46](https://github.com/tellae/eqasim/commit/3457a468443e0cfb0d848ad6bb07f366570081ed))
* Update Levenshtein dependency ([#134](https://github.com/tellae/eqasim/issues/134)) ([1aecebb](https://github.com/tellae/eqasim/commit/1aecebb25f1d0dcfd4e332f7f5c8578eb146ff97))
* Update to BPE 2021 ([#130](https://github.com/tellae/eqasim/issues/130)) ([1d797a6](https://github.com/tellae/eqasim/commit/1d797a67ae1743d1b82c6f2620c3da5a4f08a145))

## Changelog

**Under development**

- Fix: Arbitrary order of week days in merged GTFS
- Use BPE 2021 instead of BPE 2019
- Update configuration files for Lyon, Nantes, Corsica
- Add a basic sample based vehicle fleet generation tailored for use with the `emissions` matsim contrib
- Fixing socioprofessional category for Nantes and Lyon (Cerema)
- Fix documentation and processing for Nantes GTFS
- Add law status of the SIRENE enterprises for down-stream freight models (this requires both SIREN and SIRET data as input!)
- Update handling of invalid values on the nubmer of employees in SIRENE
- Add alternative source for EDGT Lyon (and set it as default/recommended source)
- Add euclidean distance to Nantes/Lyon GTFS output
- Fix GTFS schedules without transfer times
- Added stage to write out the full merged GTFS feed: `data.gtfs.output`
- Bugfix: Sometimes bug in converting GTFS coordinates (esp. Lyon / Nantes)
- Fixing output stages
- Add output stages for SIRENE and the selected HTS
- Add output prefix to non-MATSim output files as well
- Add code and documentation for Nantes use case
- Bugfix: Generate `meta.json` when code was not cloned but downloaded directly
- Use `eqasim-java:1.3.1`
- Make choice of branch and version of pt2matsim more flexible
- Improve handling of Osmosis on Windows
- Add stages to process EDGT for Lyon

**1.2.0**

- Update code and data to BPE 2019 (verison for 2018 is not available anymore)
- Add additional spatial standard output: `homes.gpkg` and `commutes.gpkg`
- Updated documentation for BD-TOPO
- By default, load SIRENE directly from `zip` file instead of `csv`
- Bugfix: Make sure df_trips are sorted properly in `synthesis.population.trips`
- Bugfix: Do not execute "urban" attribute imputation twice
- Bugfix: Do not consider *inactive* enterprises from SIRENE
- Update analysis scripts
- Remove CRS warnings
- Bugfix: Handle case if very last activity chain in population ends with tail
- Speed up and improve testing
- Improve analysis output for ENTD
- Update to `eqasim-java:1.2.0` to support tails and "free" activity chains
- Allow for activity chains that do not start and end at home
- Improve handling of education attribute in ENTD

**1.1.0**

- Update to `synpp:1.3.1`
- Use addresses for home locations (from BD-TOPO)
- Use enterprise addresses for work locations (from SIRENE + BD-TOPO)
- Add SIRENE and BD-TOPO data sets
- Update to `eqasim-java:1.1.0` and MATSim 12
- Preparation to use Corisca scenario (see config_corsica.yml) as unit test input in `eqasim-java`
- Several auto-fixes for malformatted GTFS schedules (mainly Corsica)
- Make jar output optional and use proper prefix
- Bugfix: Fixing bug where stop times where discarded in GTFS cutting
- Add documentation for Lyon and Toulouse
- Define stage to output HTS reference data
- Make prefix of MATSim output files configurable
- Cut GTFS schedules to the scenario area automatically
- Make possible to merge multiple GTFS files automatically
- Automatically convert, filter and merge OSM data before using it in pt2matsim. This requires that `osmosis` is available in the run environment.
- Provide calibrated Île-de-France/Paris eqasim simulation for 5% sample
- Make use of `isUrban` attribute from eqasim `1.0.6`
- Update to eqasim `1.0.6`
- Make GTFS date configurable
- Use synpp 1.2.2 to fix Windows directory regeneration issue
- Make pipeline configurable for other departments and regions, add documentation
- BC: Make use of INSEE zone summary data (`codes_2017`)
- Add configuration parameters to filter for departments and regions
- Fixed destinations that have coordinates outside of their municipality
- Make error message for runtime dependencies more verbose
- Switch default instructions to Anaconda

**1.0.0**

- Fixed dependency issue for ENTD scenario
- Initial public version of the pipeline
