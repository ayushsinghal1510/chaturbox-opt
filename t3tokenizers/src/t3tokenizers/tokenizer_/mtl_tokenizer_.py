import torch

from torch import Tensor
from tokenizers import Tokenizer

from .services_ import SERVICES 

class MTLTokenizer(SERVICES) : 

    def __init__(self , config : dict) -> None : 

        super().__init__(config)

        vocab_file_path : str = config['vocab-file-path']

        self.tokenizer : Tokenizer = Tokenizer.from_file(vocab_file_path)

        self.config : dict = config

        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self) -> None : 

        vocab = self.tokenizer.get_vocab()

        sot_token : str = self.config['tokens']['sot']
        eot_token : str = self.config['tokens']['eot']

        assert sot_token in vocab , f'SOT Token : {sot_token} | not in vocab'
        assert eot_token in vocab , f'EOT Token : {eot_token} | not in vocab'

    def text_to_tokens(
        self , 
        text : str , 
        language_id : str = 'en' , 
        lowercase : bool = True , 
        nfkd_normalize : bool = True
    ) : 

        text_tokens = self.encode(
            self.pre_process_text(text) , 
            language_id = language_id , 
            lowercase = lowercase , 
            nfkd_normalize = nfkd_normalize
        )

        text_tokens : Tensor = torch.IntTensor(text_tokens).unsqueeze(0)

        return text_tokens

    def encode(
        self , 
        txt : str , 
        language_id : str = 'en' , 
        lowercase : bool = True , 
        nfkd_normalize : bool = True
    ) : 

        txt = self.pre_process_text(
            text = txt , 
            lowercase = lowercase , 
            nfkd_normalize = nfkd_normalize
        )
        
        # # Language-specific text processing
        # if language_id == 'zh':
        #     txt = self.cangjie_converter(txt)
        # elif language_id == 'ja':
        #     txt = hiragana_normalize(txt)
        # elif language_id == 'he':
        #     txt = add_hebrew_diacritics(txt)
        # elif language_id == 'ko':
        #     txt = korean_normalize(txt)
        # elif language_id == 'ru':
        #     txt = add_russian_stress(txt)
        
        # Prepend language token
        if language_id : 
            txt = f'[{language_id.lower()}]{txt}'
        
        txt = txt.replace(' ', self.config['tokens']['space'])

        return self.tokenizer.encode(txt).ids

    def decode(self , seq) : 

        if isinstance(seq , Tensor) : 
            seq = seq.cpu().numpy()

        txt = self.tokenizer.decode(seq , skip_special_tokens = False)

        txt = txt.replace(
            ' ' , ''
        ).replace(
            self.config['tokens']['space'] , ' '
        ).replace(
            self.config['tokens']['eot'] , ''
        ).replace(
            self.config['tokens']['unk'], ''
        )

        return txt
