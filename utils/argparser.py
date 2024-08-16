import argparse

def arg_parse():

    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument('-t', "--train", dest="train",action="store_true", help="Weather to train the CNN")
    parser.add_argument('-e',  '--epochs',               type=int,   required=False, default = 5, help="Number of epochs for training")
    parser.add_argument('-lr', '--learning_rate',        type=float, required=False, default = 0.001, help="Learning rate for training")

    # Parse the arguments
    return parser.parse_args()
