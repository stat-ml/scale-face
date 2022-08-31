import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def pretty_matplotlib_config(fontsize=15):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams.update({'font.size': fontsize})


def main():
    print("hello")
    fig = plt.figure(figsize=(9, 7))
    pretty_matplotlib_config(22)
    s = np.arange(0, 1.01, 0.02)
    sigma = 0.4
    for a in [0.4, 0.6, 0.8]:
        phi = norm.cdf((s/sigma*(1-a)))
        y = 2*(1 - phi)
        label = f"a={a}"
        plt.plot(s, y, label=label, linewidth=3)

    plt.xlabel("Scale")
    plt.ylabel("Error probability")
    plt.legend()
    fig.savefig('tmp/theory_fig.pdf', dpi=150, format='pdf')

if __name__ == '__main__':
    main()