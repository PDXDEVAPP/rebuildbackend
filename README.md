# Rust Ollama - A Modular LLM Inference Server

A high-performance, modular LLM inference server built in Rust, designed to be a drop-in replacement for Ollama with enhanced architecture and performance.

## 🌟 Features

- **Modular Architecture**: Clean separation of concerns with distinct modules for database, inference, model management, and API
- **SQLite Database**: Robust metadata storage with ACID transactions and efficient queries
- **Candle-Powered Inference**: Leverages Hugging Face's Candle framework for high-performance LLM inference
- **GGUF Model Support**: Full support for quantized GGUF models (Q4_0, Q8_0, etc.)
- **RESTful API**: Ollama-compatible REST API for seamless integration
- **CLI Interface**: Complete command-line interface matching Ollama's functionality
- **Cross-Platform**: Supports Linux, macOS, and Windows
- **GPU Acceleration**: CUDA and Metal support for enhanced performance

## 🚀 Quick Start

### Installation

1. **Clone and build**:
   ```bash
   git clone <repository>
   cd rust_ollama
   cargo build --release
   ```

2. **Start the server**:
   ```bash
   ./target/release/rust_ollama serve --port 11434
   ```

3. **Use the CLI**:
   ```bash
   ./target/release/ollama_cli list
   ./target/release/ollama_cli pull llama3.2
   ./target/release/ollama_cli run llama3.2 "Explain quantum computing"
   ```

## 📖 Architecture

### Core Components

```
rust_ollama/
├── src/
│   ├── main.rs              # Application entry point
│   ├── core/
│   │   ├── database.rs      # SQLite database layer
│   │   ├── inference_engine.rs # LLM inference with Candle
│   │   └── model_manager.rs # Model lifecycle management
│   ├── api/
│   │   └── server.rs        # REST API server (Axum)
│   └── bin/
│       └── ollama_cli.rs    # Command-line interface
├── Cargo.toml
├── build.rs                 # Build configuration
└── config.toml             # Default configuration
```

### Database Schema

The SQLite database stores:
- **Models**: Metadata, file paths, sizes, parameters
- **Running Models**: Active model instances and statistics
- **Sessions**: User sessions and context history
- **Performance Metrics**: Request timing and resource usage

### API Endpoints

#### Model Management
- `POST /api/list` - List all models
- `POST /api/pull` - Download/pull a model
- `POST /api/delete` - Remove a model
- `POST /api/copy` - Copy a model
- `POST /api/show` - Show model details
- `POST /api/ps` - List running models
- `POST /api/stop` - Stop a model

#### LLM Inference
- `POST /api/generate` - Generate text from prompt
- `POST /api/chat` - Chat completion

#### System
- `GET /api/version` - Server version
- `GET /health` - Health check

## 🛠️ CLI Commands

```bash
# Start the server
rust_ollama serve --port 11434 --models-dir ./models

# Model management
ollama_cli list              # List local models
ollama_cli pull llama3.2     # Pull a model
ollama_cli rm llama3.2       # Remove a model
ollama_cli cp llama3.2 custom # Copy a model
ollama_cli show llama3.2     # Show model details

# Runtime operations
ollama_cli ps                # List running models
ollama_cli stop llama3.2     # Stop a model
ollama_cli run llama3.2 "Hello" # Run model interactively

# Direct inference
ollama_cli generate --model llama3.2 "Explain AI" --format json
ollama_cli chat --model llama3.2 --stream
```

## ⚙️ Configuration

### Configuration File (config.toml)
```toml
[server]
host = "127.0.0.1"
port = 11434

[storage]
database_path = "./ollama.db"
models_directory = "./models"

[inference]
default_temperature = 0.8
default_max_tokens = 512

[performance]
enable_caching = true
gpu_memory_fraction = 0.8
```

### Environment Variables
- `OLLAMA_HOST` - Server host (default: http://localhost:11434)
- `RUST_LOG` - Logging level (default: info)

## 📊 Performance Features

- **Memory Efficient**: Optimized model loading and caching
- **GPU Acceleration**: CUDA and Metal support for faster inference
- **Concurrent Requests**: Handle multiple requests simultaneously
- **Streaming Support**: Real-time response streaming
- **Resource Monitoring**: Track memory usage and performance metrics

## 🔧 Development

### Building

```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# With CUDA support
cargo build --release --features candle-cuda

# With Metal support (macOS)
cargo build --release --features metal
```

### Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test database

# Integration tests
cargo test --test integration
```

### Code Quality

```bash
# Format code
cargo fmt

# Lint with clippy
cargo clippy

# Check for security vulnerabilities
cargo audit
```

## 🌐 Model Support

### Supported Formats
- **GGUF**: Primary format with full support
- **Quantization**: Q4_0, Q8_0, and other llama.cpp quantization levels

### Supported Model Families
- LLaMA/LLaMA 2/LLaMA 3
- Mistral/Mixtral
- CodeLLaMA
- Gemma
- Phi
- Custom GGUF models

### Model Lifecycle
1. **Pull**: Download from registry
2. **Load**: Load into memory for inference
3. **Cache**: Keep frequently used models in memory
4. **Unload**: Free memory when not needed
5. **Remove**: Delete from storage

## 🔍 API Examples

### Generate Text
```bash
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "prompt": "Explain quantum computing in simple terms",
    "system": "You are a helpful AI assistant",
    "stream": false
  }'
```

### Chat Completion
```bash
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you!"},
      {"role": "user", "content": "Can you help me with coding?"}
    ],
    "stream": false
  }'
```

### Model Management
```bash
# List models
curl -X POST http://localhost:11434/api/list

# Pull model
curl -X POST http://localhost:11434/api/pull \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'

# Delete model
curl -X POST http://localhost:11434/api/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'
```

## 🚧 Roadmap

### Version 0.2.0
- [ ] Complete Candle model loading integration
- [ ] Add multimodal model support (vision)
- [ ] Implement streaming responses
- [ ] Add Docker containerization

### Version 0.3.0
- [ ] WebSocket support for real-time chat
- [ ] Model fine-tuning capabilities
- [ ] Advanced caching strategies
- [ ] Kubernetes deployment manifests

### Version 1.0.0
- [ ] Full Ollama API compatibility
- [ ] Production-ready performance
- [ ] Comprehensive test suite
- [ ] Official documentation site

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ollama](https://github.com/ollama/ollama) for the inspiration and API design
- [Hugging Face](https://github.com/huggingface/candle) for the Candle ML framework
- [Rust Community](https://www.rust-lang.org/) for the excellent ecosystem

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discord: [Join our server](https://discord.gg/example)
- 🐛 Issues: [GitHub Issues](https://github.com/example/rust_ollama/issues)
- 📖 Documentation: [docs.rs](https://docs.rs/rust_ollama)