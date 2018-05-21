import matplotlib.pyplot as plt
import silhuetaDInamico as sd
import numpy as np

def plotar_silhueta(silhueta_dados):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    Y = np.arange(0, len(silhueta_dados))
    ax.fill_betweenx(Y, 0, silhueta_dados)
    plt.plot()

if __name__ == "__main__":
    
    pass



