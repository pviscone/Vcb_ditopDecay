import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import mplhep
import sys

plt.style.use(mplhep.style.CMS)
class Histogrammer():
    def __init__(self, xlabel="", ylabel="Counts", title="", cmsText="Preliminary", legendloc="best",
                 fontsize=20,legend_fontsize=13,
                 histrange=None,histtype="stepfilled",linewidth=2,
                 legend=True, mean=True, rms=True, N=False, total_stats=False,
                 log=False, grid=False,  xlim=None, ylim=None,
                 **kwargs):
        self.hist_list = []
        self.range=histrange
        self.histtype=histtype
        self.linewidth=linewidth
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.log = log
        self.cmsText = cmsText
        self.log = log
        self.grid = grid
        self.legend = legend
        self.mean = mean
        self.rms = rms
        self.N=N
        self.xlim = xlim
        self.ylim = ylim
        self.legendloc = legendloc
        self.total_stats=total_stats
        self.common_kwargs = kwargs
        
        

        plt.rc("font", size=fontsize)
        plt.rc('legend', fontsize=legend_fontsize)


    def add_hist(self, array, label="",**kwargs):
        

        if "range" in kwargs:
            histrange=kwargs["range"]
            del kwargs["range"]
        else:
            histrange=self.range
            
        if "histtype" in kwargs:
            histtype=kwargs["histtype"]
            del kwargs["histtype"]
        else:
            histtype=self.histtype
            
        if "linewidth" in kwargs:
            linewidth = kwargs["linewidth"]
            del kwargs["linewidth"]
        else:
            linewidth = self.linewidth
            
        common_kwargs = self.common_kwargs
        for key in kwargs.keys():
            if (key in common_kwargs):
                del common_kwargs[key]

        if histrange == None:
            new_array = array
        else:
            new_array = array[np.bitwise_and(
                array > histrange[0], array < histrange[1])]
        legend_label = label+" "
        if self.N:
            legend_label += f" (N={len(new_array)})"
        legend_label += "\n"
        if self.mean:
            legend_label += f"{np.mean(new_array):.2f}"
        if self.rms:
            legend_label += f" ({np.std(new_array):.2f})"
        
        if self.mean or self.rms:
            split = (self.xlabel).split("[")
            if len(split)>1:
                legend_label += f' [{(self.xlabel).split("[")[-1]}'
        

        if self.total_stats:
            legend_label += f"\n"
            if self.mean:
                legend_label += f" $\mu_{{Tot}}$={np.mean(array):.2f}"
            if self.rms:
                legend_label += f" $\sigma_{{Tot}}$={np.std(array):.2f}"
            if self.N:
                legend_label += f" $N_{{Tot}}$={len(array)}"



            
        self.hist_list.append(plt.hist(new_array, label=legend_label, range=histrange, histtype=histtype,linewidth=linewidth, **kwargs, **common_kwargs))
    

    def plot(self):


        max_hist=-sys.maxsize
        for hist in self.hist_list:
            hist
            if(np.max(hist[0])>max_hist):
                max_hist=np.max(hist[0])
        
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        if(type(self.log)==str):
            if(self.log.lower().find("y") != -1):
                plt.yscale("log")
            if(self.log.lower().find("x") != -1):
                plt.xscale("log")
        if(self.grid):
            plt.grid()
        if(self.legend):
            plt.legend(loc=self.legendloc)
        if(self.cmsText):
            mplhep.cms.text(self.cmsText)
            
        plt.xlim(self.range[0],self.range[1])
        if(self.ylim):
            plt.ylim(self.ylim)
        else:
            if(type(self.log)==str):
                if(self.log.lower().find("y") != -1):
                    plt.ylim(0.1, max_hist*8)
            else:
                plt.ylim(0, max_hist*1.4)
        

class Subplots():
    def __init__(self,shape,figsize=(6.4,4.8),dpi=100,fontsize=20,**kwargs):

        self.figsize=figsize
        self.dpi=dpi
        self.fontsize=fontsize
        self.shape=shape
        self.kwargs=kwargs
        self.subplots = []
        
        for i in range(np.prod(shape)):
            self.subplots.append(Histogrammer(**kwargs))


    def add_subplot(self,subplot,array,**kwargs):
        plt.subplot(*self.shape,subplot)
        self.subplots[subplot-1].add_hist(array, **kwargs)
        

    def plot(self):
        plt.figure(figsize=self.figsize,dpi=self.dpi)
        plt.rc("font", size=self.fontsize)
        for idx,subplot in enumerate(self.subplots):
            plt.subplot(*self.shape,idx+1)
            subplot.plot()
        plt.show()
        



#testing
if __name__=="__main__":
    #plt.figure(figsize=(20,20))
    
    a1=np.random.normal(0, 1, 1000)
    a2=np.random.normal(-1, 1, 1000)
    a3=np.random.normal(1, 1, 1000)
    a4=np.random.normal(2, 1, 1000)
    a5=np.random.normal(3, 1, 1000)
    a6=np.random.normal(4, 1, 1000)
    a7=np.random.normal(5, 1, 1000)
    a8=np.random.normal(6, 1, 1000)
    """ plt.subplot(2,2,1)
    h1 = Histogrammer(log="y", xlabel="asd [Gev]",bins=100)
    h1.add_hist(a1, label="test")
    h1.add_hist(a2, label="test2",bins=5)
    h1.plot()

    
    plt.subplot(2,2,2)
    h2 = Histogrammer(log="y", xlabel="asd [Gev]")
    h2.add_hist(a3, label="test")
    h2.add_hist(a4, label="test2")
    h2.plot()
    
    plt.subplot(2,2,3)
    h3 = Histogrammer(log="y", xlabel="asd [Gev]")
    h3.add_hist(a5, label="test")
    h3.add_hist(a6, label="test2")
    h3.plot()
    
    plt.subplot(2,2,4)
    h4 = Histogrammer(xlabel="asd [Gev]")
    h4.add_hist(a7, label="test")
    h4.add_hist(a8, label="test2")
    h4.plot()
    plt.show() """

     
    h = [Histogrammer(log="y", xlabel="asd [Gev]", bins=100)]*4
    plt.figure(figsize=(5,10))
    plt.subplot(1, 2, 1)
    h[1].add_hist(a1, label="test")
    h[1].add_hist(a2, label="test2", bins=5)
    h[1].plot()
    
    plt.subplot(1, 2, 2)
    h[2].add_hist(a3, label="test")
    h[2].add_hist(a4, label="test2",bins=100)
    h[2].plot()

    
    plt.figure(figsize=(5,10))
    one=np.ones(10)
    h=Histogrammer(log="y", xlabel="asd [Gev]", bins=100)
    h.add_hist(one, label="test")
    h.plot()
    plt.show()
