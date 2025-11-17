# NVDB-RS: Nearest Vector Database in Rust

## Overview
NVDB-RS is a Rust-based library designed to handle nearest vector searches efficiently. It is a work in progress and currently provides functionalities for managing segments, storing vectors, and performing k-nearest neighbor (k-NN) searches using different distance metrics.

## Features
- **Segment Storage**:
  - Write and read vector data to/from segments.
  - Automatically generates metadata files (`meta.json`).

- **k-NN Indexing**:
  - Supports multiple metrics, including Euclidean, Cosine similarity, and Dot product.
  - Allows for efficient vector search and result ranking.

- **Metric Implementations**:
  - `Euclidean` for distance-based similarity.
  - `Cosine` for angle-based similarity.
  - `Dot Product` for projection-based calculations.

## Getting Started
### Prerequisites
- Rust programming language (>= 1.65)

### Installation
To use NVDB-RS in your project, add it as a dependency in your `Cargo.toml` file:
```toml
[dependencies]
nvdb-rs = "0.1.0"
```

## Tests
The project includes comprehensive tests to ensure stability and correctness:
- **Segment Tests**:
  - Writes vector data to segments and verifies data consistency.
  - Reads and indexes data for search operations.

- **Index Tests**:
  - Validates k-NN searches using various distance metrics.

- **Metrics Tests**:
  - Ensures accurate calculations for Euclidean, Cosine, and Dot Product scores.

### Running Tests
Run the tests using Cargo:
```bash
cargo test
```

## Contributions
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
Thanks to the open-source Rust community for their valuable crates and tools.

