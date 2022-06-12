from textblob import TextBlob
import language_tool_python


class Grammar_checker:

    def __init__(self) -> None:
        self.my_tool = language_tool_python.LanguageTool('en-US')

    def genetate_new_sentence(self, sentence, start_positions, 
            sntc_corrections, end_positions) -> str:
        """
            Method to generate a new sentence according to the right
            corrections.
        """
        
        new_sntc = list(sentence)
        for n in range(len(start_positions)):
            for i in range(len(sentence)):
                new_sntc[start_positions[n]] = sntc_corrections[n]
                if (i > start_positions[n] and i < end_positions[n]):
                    new_sntc[i] = ""
        
        return "".join(new_sntc)
    
    def correct_word_sintax(self, sentence) -> str:
        """
            Method to check and correct the word sintax.
        """
        textBlb = TextBlob(sentence)
        return str(textBlb)

    def check_sentence(self, sentence) -> str:
        """
            Method that will correct a given sentence 
            base on the sintax and gammatical erros.
        """
        sntc_matches = self.my_tool.check(sentence)
        sntc_mistakes = []
        sntc_corrections = []
        start_positions = []
        end_positions = []

        for rules in sntc_matches:
            if len(rules.replacements) > 0:
                start_positions.append(rules.offset)
                end_positions.append(rules.errorLength + rules.offset)
                sntc_mistakes.append(
                    sentence[rules.offset: rules.errorLength + rules.offset])
                sntc_corrections.append(rules.replacements[0])

        return self.genetate_new_sentence(
                    sentence, start_positions, sntc_corrections, end_positions)