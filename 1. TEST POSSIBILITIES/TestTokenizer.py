# Make sure you have the transformers and datasets library installed
# pip install transformers datasets

from transformers import WhisperTokenizer

def test_tokenizer_on_new_language(
    pretrained_model_name: str,
    tokenizer_language: str,
    target_text: str,
    target_language_name: str
):
    """
    Tests a pre-trained Whisper tokenizer on text from a new language.

    Args:
        pretrained_model_name (str): The name of the pre-trained Whisper model
                                     (e.g., 'openai/whisper-small').
        tokenizer_language (str): The language the tokenizer was originally
                                  trained/configured for (e.g., 'bengali').
        target_text (str): The sample text from the new language you want to test.
        target_language_name (str): The name of the target language (e.g., 'Odia').
    """
    print(f"Loading tokenizer for '{tokenizer_language}' from '{pretrained_model_name}'...")
    try:
        # Load the tokenizer for the 'source' language
        tokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name,
            language=tokenizer_language,
            task="transcribe"
        )
        print("Tokenizer loaded successfully.")

        print(f"\nOriginal {target_language_name} text:\n{target_text}")

        # Tokenize the target language text
        print("\nTokenizing the text...")
        tokenized_output = tokenizer(target_text)
        input_ids = tokenized_output.input_ids
        print(f"Token IDs: {input_ids}")

        # Decode the tokenized text
        print("\nDecoding the token IDs...")
        decoded_text = tokenizer.decode(input_ids)
        print(f"Decoded text:\n{decoded_text}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure the model name and language code are correct and")
        print("that you have an internet connection.")

# --- Example Usage (based on your image) ---
model = "openai/whisper-small"
source_lang_code = "spanish" # Language A (Tokenizer's language)
#target_lang_text = "Oñepyrũ ojeporeka, oporandu maymáva hapichakuéra kuépe. Heta rire, peteĩ árape, osẽ ichupe" # Language B (Odia text)
#target_lang_text = "Upepeténtevoi omopu'ã tapỹi tupemi ha oiko ipype ha'eño peteĩ. Arañavõ omomorã imemby kurusu, otupanói ha ohovasa ichupe." # Language B (Odia text)
target_lang_text = "sshekh yer vannevae rhojosores yeri ma adothrae yomme krazaajoon" # Language B (Odia text)
target_lang = "dothraki"

test_tokenizer_on_new_language(
    model,
    source_lang_code,
    target_lang_text,
    target_lang
)

print("\n" + "="*50 + "\n")

'''
# --- Example with English tokenizer and German text ---
model = "openai/whisper-base"
source_lang_code = "english"
target_lang_text = "Wie geht es Ihnen?" # German text
target_lang = "German"

test_tokenizer_on_new_language(
    model,
    source_lang_code,
    target_lang_text,
    target_lang
)
'''