
from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider

class LLMProviderFactory:
    def __init__(self, config: dict, ):
        self.config = config

    def create(self, provider: str, embedding: bool = False):
        print(f"Creating LLM provider: {provider}")
        print(f"Embedding: {embedding}")
        # print(f"Config: {self.config.OPENAI_API_EMBEDDING_URL}")
        if provider == LLMEnums.OPENAI.value:
            # if embedding:
            #     return OpenAIProvider(
            #         api_key=self.config.OPENAI_API_KEY,
            #         api_url=self.config.OPENAI_API_EMBEDDING_URL,
            #         default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
            #         default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
            #         default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            #     )
            # else:
                  return OpenAIProvider(
                    api_key=self.config.OPENAI_API_KEY,
                    api_url=self.config.OPENAI_API_URL,
                    default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                    default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                    default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
                )


        if provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key = self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        return None
