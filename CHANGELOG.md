# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 2026-04-06

* 🔵 [dev] Initial release of Neurobridge V1 pipeline. 
* 🔵 [dev] Integrated 8-phase architecture including ingestion, signal processing, and medical label alignment.
* 🔵 [dev] Built Phase 7 Actuation Interface bridging neural decoding to simulated and TCP hardware commands.
* 🔵 [dev] Built Phase 9 Latency Optimization, replacing predict overhead with direct tensor invocation for microsecond real-time inference.
* 🔵 [ops] Cleaned git tracking to exclude massive binary clinical datasets (30GB+) ensuring a lightweight production repository.