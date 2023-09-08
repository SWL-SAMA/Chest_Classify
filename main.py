import Building
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()
# Building.train(args.epochs)
Building.test()