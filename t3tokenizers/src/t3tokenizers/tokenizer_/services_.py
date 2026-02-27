from unicodedata import normalize

class SERVICES : 

    def __init__(
        self , 
        config : dict
    ) -> None : 

        self.config : dict = config

    def pre_process_text(
        self , 
        text : str , 
        lowercase : bool = True , 
        nfkd_normalize : bool = True
    ) -> str : 

        assert (
            text and 
            len(text) > 0 
        ) , 'Text cannot be empty'

        text = text.strip()

        if lowercase : 
            text = text.lower()

        if nfkd_normalize : 
            text = normalize('NFKD' , text)

        text = text[0].upper() + text[1:]

        # punctuations : dict[str , str] = self.config['pre-process']['punctuations']

        for row in self.config['pre-process']['punctuations'] : 
            text = text.replace(row[0] , row[1])

        last_charac : str = text[-1]

        if last_charac not in self.config['pre-process']['allowed-endings'] : 
            text = text[:-1] + self.config['pre-process']['default-ending']

        return text