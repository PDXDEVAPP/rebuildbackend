use clap::Parser;
use std::path::PathBuf;
use tracing::{info, error};

// Define modules
mod core {
    pub mod database;
    pub mod inference_engine;
    pub mod model_manager;
}

mod api {
    pub mod server;
}

#[derive(Parser)]
#[command(name = "rust_ollama")]
#[command(about = "A modular Rust-based LLM inference server with Ollama-compatible API")]
#[command(version = "0.1.0")]
#[command(author = "MiniMax Agent")]
struct Args {
    /// Database file path
    #[arg(short, long, default_value = "./ollama.db")]
    database: String,
    
    /// Models directory
    #[arg(short, long, default_value = "./models")]
    models_dir: String,
    
    /// Server port
    #[arg(short, long, default_value = "11434")]
    port: u16,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Run in CLI mode (instead of server mode)
    #[arg(long)]
    cli: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt::init();
    } else {
        env_logger::init();
    }
    
    info!("Starting Rust Ollama server v{}", env!("CARGO_PKG_VERSION"));
    info!("Database: {}", args.database);
    info!("Models directory: {}", args.models_dir);
    info!("Port: {}", args.port);
    
    if args.cli {
        // Run CLI mode - delegate to the CLI binary
        let cli_args = std::env::args().collect::<Vec<_>>();
        let cli_program = "ollama_cli";
        let cli_path = std::env::current_exe()?
            .parent()
            .unwrap_or(std::env::current_dir()?)
            .join(cli_program);
        
        let mut command = tokio::process::Command::new(&cli_path);
        command.args(&cli_args[1..]); // Skip the program name
        
        let status = command.spawn()?.wait().await?;
        
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
        
        return Ok(());
    }
    
    // Initialize core components
    let database = crate::core::database::DatabaseManager::new(&args.database).await?;
    info!("Database initialized successfully");
    
    let inference_engine = crate::core::inference_engine::InferenceEngine::new();
    info!("Inference engine initialized");
    
    let model_manager = crate::core::model_manager::ModelManager::new(
        database,
        inference_engine,
        PathBuf::from(&args.models_dir),
    );
    
    // Initialize model manager
    model_manager.initialize().await?;
    info!("Model manager initialized successfully");
    
    // Start API server
    let api_server = crate::api::server::ApiServer::new(model_manager);
    info!("Starting API server on http://127.0.0.1:{}", args.port);
    
    if let Err(e) = api_server.start(args.port).await {
        error!("Failed to start API server: {}", e);
        return Err(e.into());
    }
    
    Ok(())
}