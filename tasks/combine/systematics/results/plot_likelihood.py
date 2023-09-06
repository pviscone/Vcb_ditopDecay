import uproot
import glob
import matplotlib.pyplot as plt
import numpy as np



limit=uproot.open("combine.root")["limit"]

r=limit["r"].array()
sort=np.argsort(r)
r=r[sort]
NLL=2*limit["deltaNLL"].array()[sort]

plt.plot(r,NLL)
rmin=r[r<1][np.argmin(np.abs(NLL[r<1]-1))]
rmax=r[r>1][np.argmin(np.abs(NLL[r>1]-1))]
print(f"{rmin:.2f} < r < {rmax:.2f} ({(rmax-1)*100:.2f}%)")

plt.plot([0.8,1.2],[1,1],color="red")
plt.legend()
plt.grid()

plt.savefig("plot.pdf")
