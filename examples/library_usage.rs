use rust_ollama::core::database::DatabaseManager;
use rust_ollama::core::inference_engine::{InferenceEngine, InferenceConfig};
use rust_ollama::core::model_manager::ModelManager;
use std::path::PathBuf;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    info!("Starting Rust Ollama Example");
    
    // Initialize database
    let database = DatabaseManager::new("./example.db").await?;
    info!("Database initialized");
    
    // Initialize inference engine
    let inference_engine = InferenceEngine::new();
    info!("Inference engine initialized");
    
    // Initialize model manager
    let models_dir = PathBuf::from("./example_models");
    let model_manager = ModelManager::new(database, inference_engine, models_dir);
    
    // Initialize model manager
    model_manager.initialize().await?;
    info!("Model manager initialized");
    
    // List local models
    let models = model_manager.list_local_models().await?;
    info!("Found {} local models", models.len());
    
    for model in models {
        info!("Model: {} ({:?})", model.name, model.model_type);
    }
    
    // Search for models
    let search_results = model_manager.search_local_models("llama").await?;
    info!("Search results for 'llama': {} models found", search_results.len());
    
    // Get model statistics
    let stats = model_manager.get_model_stats().await?;
    info!("Model statistics:");
    info!("  Total models: {}", stats.total_models);
    info!("  Running models: {}", stats.running_models);
    info!("  Total size: {:.2} GB", stats.total_size_gb());
    
    println!("\n🎉 Example completed successfully!");
    println!("Database file: ./example.db");
    println!("Models directory: ./example_models");
    
    Ok(())
}