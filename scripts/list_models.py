import os
import logging
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """List available Gemini models and their supported methods."""
    # Configure Google GenAI with API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return
    
    genai.configure(api_key=api_key)
    
    try:
        # List available models
        logger.info("Listing available Gemini models...")
        models = genai.list_models()
        
        # Filter for embedding models only
        logger.info("Models supporting embedContent:")
        embedding_models = [m for m in models if 'embedContent' in m.supported_generation_methods]
        
        if not embedding_models:
            logger.info("No models found that support embedContent")
            
            # Show all available models for reference
            logger.info("\nAll available models and their supported methods:")
            for model in models:
                logger.info(f"- {model.name}")
                logger.info(f"  Supported methods: {model.supported_generation_methods}")
        else:
            # Show detailed info for embedding models
            for model in embedding_models:
                logger.info(f"- {model.name}")
                logger.info(f"  Display name: {model.display_name}")
                logger.info(f"  Description: {model.description}")
                logger.info(f"  Supported generation methods: {model.supported_generation_methods}")
                logger.info("")
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    main() 