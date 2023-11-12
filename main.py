from settings import *
from utils import create_model, predict_sprod, plot_final_results


def main():
    create_model()
    plot_final_results(MODELS)
    sprod = predict_sprod(MODELS)
    return sprod

if __name__ == '__main__':
    main()