import torch

from torch import Tensor

from transformers import (
    LlamaConfig , 
    LlamaModel , 
    LlamaPreTrainedModel , 
    GenerationMixin
)

from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from ..alignment_stream_analyzer import AlignmentStreamAnalyzer


class T3HuggingfaceBackend(LlamaPreTrainedModel , GenerationMixin) : 

    '''
    Override some HuggingFace interface methods so we can use the standard `generate` method with our
    custom embedding / logit layers.

    NOTE: need to extend '*PreTrainedModel' to avoid re-initializing weights!

    Args : 
        - config : LlamaConfig : Config for the llama model to load, like dimensions, how much heads etc
        - llama : LlamaModel : Defines which model to use and put the config, when put the llama config in this model instance, it basically gives the llama model

        - speech_enc : S3 Tokenizer being used for tokenizing audio 
        - speech_head : Custom head to be replaced with lm_head 
        - latents_queue
    '''

    def __init__(
        self , 
        config : LlamaConfig ,
        llama : LlamaModel , 
        * , 
        speech_enc , 
        speech_head , 
        latents_queue = None , 
        logits_queue = None , 
        alignment_stream_analyzer : AlignmentStreamAnalyzer | None = None
    ) : 
        super().__init__(config)
        self.model = llama
        self.speech_enc = speech_enc
        self.speech_head = speech_head
        self._added_cond = False
        self.alignment_stream_analyzer = alignment_stream_analyzer

    @torch.inference_mode()
    def prepare_inputs_for_generation(
        self , 
        input_ids : Tensor , 
        decoder_cond : Tensor , 
        use_cache : bool , 
        past_key_values = None , 
        # This argument was introduced in some recent version of transformers (>=4.29.1)
        cache_position = None
    ) : 
        """
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom speech token embedding layer.

        :param input_ids: (B, S) int64 tensors of input tokens.
        :param decoder_cond: (B, T, C) float32 tensor of conditioning (prefixed to <input_embeds>)
        """

        # Make use of the kv cache: only the last input ID is new, we trim away all the ones before

        if not use_cache : 
            past_key_values = None

        # * If the past key values are not none, take the last value out of them 
        if past_key_values is not None : 
            input_ids = input_ids[: , -1:]

        # * custom speech token embedding layer
        inputs_embeds : Tensor = self.speech_enc(input_ids)

        # * prefix decoder conditioning if applicable
        # * Only add on the first iteration, else past_key_values would already have the information
        if not self._added_cond : 

            assert past_key_values is not None # should be first step

            if decoder_cond.size(0) != inputs_embeds.size(0) : 
                decoder_cond = decoder_cond.expand(inputs_embeds.size(0) , -1 , -1)

            inputs_embeds = torch.cat([decoder_cond , inputs_embeds] , dim = 1)
            self._added_cond = True

        return {
            'inputs_embeds' : inputs_embeds , 
            'past_key_values' : past_key_values , 
            'use_cache' : use_cache
        }

    @torch.inference_mode()
    def forward(
        self , 
        inputs_embeds : Tensor , 
        past_key_values : Tensor | None = None , 
        use_cache : bool = True , 
        output_attentions : bool = False , 
        output_hidden_states : bool = True , 
        return_dict : bool = True
    ) -> CausalLMOutputWithCrossAttentions : 

        '''
        This is a method used by huggingface's generate() method.
        Overridden here to apply our custom layer norm and speech logit projection layers.

        :param inputs_embeds: (B, S, C) float32 tensor of conditioning inputs. If past key values are given,
        S should be 1.
        '''

        is_large_input = inputs_embeds.size(1) != 1
        has_cache = past_key_values is not None and len(past_key_values) > 0

        assert not (is_large_input and has_cache)
        assert return_dict
        assert output_hidden_states

        tfmr_out = self.model(
            inputs_embeds = inputs_embeds , 
            past_key_values = past_key_values , 
            use_cache = use_cache , 
            output_attentions = output_attentions , 
            output_hidden_states = output_hidden_states , 
            return_dict = True
        )

        hidden_states = tfmr_out.hidden_states[-1]  # (B, seq, dim)

        logits = self.speech_head(hidden_states)

        # assert inputs_embeds.size(0) == 1 # (disabled for CFG)

        # NOTE: hallucination handler may modify logits to force emit an EOS token
        # logits = self.alignment_stream_analyzer.step(logits)

        return CausalLMOutputWithCrossAttentions(
            logits = logits , 
            past_key_values = tfmr_out.past_key_values ,
            hidden_states = tfmr_out.hidden_states ,
            attentions = tfmr_out.attentions
        )
