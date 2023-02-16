import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt

class Histogrammer():
    def __init__(self):
        self.hist_list = []
        self.fig_properties = []
        backend_ = mpl.get_backend()
        mpl.use("Agg")  # Prevent showing stuff on screen

    def add_hist(self, array, label="", histrange=None, mean_in_legend=False, rms_in_legend=False, N_in_legend=False, xlabel="", ylabel="", title="", cmsText="Preliminary", log=False, grid=True, legend=True, **kwargs):

        property_dict = {}
        property_dict["xlabel"] = xlabel
        property_dict["ylabel"] = ylabel
        property_dict["title"] = title
        property_dict["log"] = log
        property_dict["grid"] = grid
        property_dict["legend"] = legend
        property_dict["cmsText"] = cmsText
        self.fig_properties.append(property_dict)

        if histrange == None:
            new_array = array
        else:
            new_array = array[np.bitwise_and(
                array > histrange[0], array < histrange[1])]
        legend_label = label
        if mean_in_legend:
            legend_label += f" $\mu$={np.mean(new_array):.2f}"
        if rms_in_legend:
            legend_label += f" $\sigma$={np.std(new_array):.2f}"
        if N_in_legend:
            legend_label += f" N={len(new_array)}"
        plt.ioff()

        self.hist_list.append(
            plt.hist(new_array, label=legend_label, range=histrange, **kwargs))
        plt.close()

    def plot(self):
        for hist, prop_dict in zip(self.hist_list, self.fig_properties):
            hist
            plt.show()


if __name__=="__main__":
    pass