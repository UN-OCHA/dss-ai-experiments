"""
Helper script to check if a given model supports masked language modeling.
"""

from typing import Optional
from transformers import AutoConfig, AutoModelForMaskedLM

def check_model_for_mlm(model_name: str) -> bool:
    """
    Check if a given model supports masked language modeling.

    Args:
        model_name: The name or path of the model to check.

    Returns:
        A boolean indicating whether the model supports masked language modeling.
    """
    try:
        # Try to load the model configuration.
        config = AutoConfig.from_pretrained(model_name)

        # Check if the model has a language modeling head.
        if hasattr(config, 'is_decoder') and not config.is_decoder:
            print(f"The model {model_name} likely supports AutoModelForMaskedLM.")

            # Try to load the model with AutoModelForMaskedLM.
            try:
                model = AutoModelForMaskedLM.from_pretrained(model_name)
                print(f"Successfully loaded {model_name} with AutoModelForMaskedLM.")
                return True
            except Exception as e:
                print(f"Error loading model with AutoModelForMaskedLM: {e}")
                return False
        else:
            print(f"The model {model_name} may not be suitable for AutoModelForMaskedLM.")
            return False
    except Exception as e:
        print(f"Error checking model: {e}")
        return False

def main(model_name: Optional[str] = None) -> None:
    """
    Main function to run the model check.

    Args:
        model_name: The name or path of the model to check. If None, uses a default model.
    """
    if model_name is None:
        model_name = "Alibaba-NLP/gte-en-mlm-base"

    check_model_for_mlm(model_name)

if __name__ == "__main__":
    main()
