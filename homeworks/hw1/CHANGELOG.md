# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [3.0.2] – 2024-06-21

### Fixed

- Make harness reads from stdin non-blocking. Fixes competitor issues
  when testing with short data inputs.

## [3.0.1] – 2024-06-20

### Changed

- Refactored `cp_pov` to better handle default arguments, environment
  variables, harness file checks, and libfuzzer exit codes.

## [3.0.0] – 2024-06-17

### Added

- Added references to parallelization environment variable `NPROC_VAL`.
- Added set of generic CP Base and Harness prefix and suffix environment
  variables to be used by all CPs.
- Added an `artifacts` sequence to the `cp_sources` entry in the
  `project.yaml`.
- Added a check to `entrypoint.sh` to issue a warning if the host setting for
  `vm.mmap_rnd_bits` is above 28.
- Added the "-v" option to `run.sh` to turn on verbose debug messages.
- Added the "-x" option to `run.sh` to force it to return the exit code
  from the internal `docker run` invocations.
- Added the "-h" option to `run.sh` to print a help menu.

### Changed

- Changed base Docker image from Ubuntu 22 to base-clang from OSS-Fuzz.
- Refactored harness to be libfuzzer compliant.
- Moved `container_scripts/cp_build` to `container_scripts/cp_build.tmpl`
  and added logic to `container_scripts/cmd_harness.sh` to replace
  CP build prefix and suffix environment variables with the set values.
- Changed `run.sh` to no longer add the `out/` prefix to the harness name
  argument in the `run_pov` command. The individual CPs will handle this
  name uniquely to their needs.
- Use `work/.gitkeep` and `out/.gitkeep` and `.gitignore` instead of
  READMEs to maintain `src`, `work`, and `out` in the repository.
- Updated `mock-cp-src` repo reference to tag v1.1.0.

## [2.0.3] – 2024-05-16

### Added

- Made GitHub actions generate and publish a docker image that is referenced
  in the `project_yaml` file in the `docker_image` field.
- Added a `.dockerignore` file to the repo.

## [2.0.2] – 2024-05-14

### Fixed

- Fixed incorrect `address` field in `project.yaml`.

## [2.0.1] – 2024-05-14

### Changed

- Switched from https to ssh in `project.yaml` file's `address` field.

## [2.0.0] – 2024-05-14

First release of the AIxCC Mock Challenge Project, which is
compliant with CP Sandbox v2.0.0.

<!-- markdownlint-disable-file MD024 -->
