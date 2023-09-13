import uproot
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


argparser = argparse.ArgumentParser(description='Plot likelihood scan')
argparser.add_argument('-i', '--input', help='root file with limit tree', type=list_of_strings)

list_files = argparser.parse_args().input

input_file = argparser.parse_args().input
for name in list_files:
    limit=uproot.open(name)["limit"]
    name=name.split(".")[0].split("/")[-1]
    r=limit["r"].array()
    sort=np.argsort(r)
    r=r[sort]
    NLL=2*limit["deltaNLL"].array()[sort]
    
    plt.plot(r,NLL,label=name)
    rmin=r[r<1][np.argmin(np.abs(NLL[r<1]-1))]
    rmax=r[r>1][np.argmin(np.abs(NLL[r>1]-1))]
    print(f"{name}: {rmin:.2f} < r < {rmax:.2f} ({(rmax-1)*100:.2f}%)")

plt.plot([0.8,1.2],[1,1],color="red")
plt.legend()
plt.grid()

plt.savefig("likelihood.png")
