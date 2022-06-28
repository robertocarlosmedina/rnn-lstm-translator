import argparse
from termcolor import colored

arg_pr = argparse.ArgumentParser()

arg_pr.add_argument(
    "-a", "--action", nargs="+", required=True,
    choices=[
        "console", "train", "test_model", "blue_score",
        "meteor_score", "count_parameters", "ter_score"
    ],
    help="Add an action to run this project"
)

arg_pr.add_argument(
    "-s", "--source", required=True,
    choices=[
        "en", "cv"
    ],
    help="Source languague for the translation"
)

arg_pr.add_argument(
    "-t", "--target", required=True,
    choices=[
        "en", "cv"
    ],
    help="Target languague for the translation"
)

args = vars(arg_pr.parse_args())


if args["source"] == args["target"]:
    print(
        colored("Error: Source languague and Target languague should not be the same.", "red", attrs=["bold"])
    )
    exit(1)


from src.lstm import Seq2Seq_Translator
from src.utils import check_dataset


check_dataset()
lstm_translator = Seq2Seq_Translator(args["source"], args["target"])


def execute_main_actions():
    """
        Function the execute the action according to the users need
    """
    actions_dict = {
        "console": lstm_translator.console_model_test,
        "train": lstm_translator.train_model,
        "test_model": lstm_translator.test_model,
        "blue_score": lstm_translator.calculate_blue_score,
        "meteor_score": lstm_translator.calculate_meteor_score, 
        "count_parameters": lstm_translator.count_hyperparameters,
        "ter_score": lstm_translator.calculate_ter
    }

    [actions_dict[action]() for action in args["action"]]


if __name__ == "__main__":
    execute_main_actions()
