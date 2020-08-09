import argparse


def main(args):
    print(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", "--a", choices=["all", "kmeans", "k++", ], required=True)
    parser.add_argument("--mode", "--m", choices=["bench", "one"], required=True)
    parser.add_argument()
    parser.add_argument("-plot", "-p", action="store_true", default=False)
    args = parser.parse_args()
    main(args)