import argparse
# import os

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "flask_api", "blue_score",
        "meteor_score", "matrix_confusion", "count_parameters", "ter_score"
    ],
    help="Add an action to run this project"
)
args = vars(arg_pr.parse_args())


from src.translator import Seq2Seq_Translator
from src.flask_api import Resfull_API


lstm_translator = Seq2Seq_Translator()


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": lstm_translator.console_model_test,
        "train": lstm_translator.train_model,
        "test_model": "lstm_translator.test_model",
        "flask_api": Resfull_API.start,
        "blue_score": "lstm_translator.calculate_blue_score",
        "meteor_score": "lstm_translator.calculate_meteor_score", 
        "count_parameters": "lstm_translator.count_hyperparameters",
        "ter_score": "lstm_translator.calculate_ter"
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
