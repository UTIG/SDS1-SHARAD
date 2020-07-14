#!/usr/bin/env python3

import os
import itertools
import math
import re
import glob
import logging

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


# TODO:
# evaluate bias between feature zncc. histogram showing offsets
# difference between corresponding zncc/non-zncc algorithms for each ifactor
# (method 0 vs 1, 2 v 3, 4 vs 5, 6 vs 7)
# difference between corresponding  adaptive/nonadaptive
# If there is bias, is it significant, is it more accurate

# Compare interferograms between methods

def organize_subdirs(rootdir):
    """ Look for subdirectories of files, and extract the parameters
    return this as a list of dicts with sub directory name, and parameter values
    """


    dirlist = [(f.path, f.name) for f in os.scandir(rootdir) if f.is_dir()]

    outlist = []
    outdict = {}
    for path, name in dirlist:
        info = {'path': path, 'name': name}
        m = re.match(r'coregmethod(\d+)_if(\d+)$', name)
        if not m:
            print("Could not match " +  name)
            continue
        info['method'] = int(m.group(1), 10)
        info['ifactor'] = int(m.group(2), 10)

        # Search for coregistration data file
        logging.info("Processing " + path)
        cfiles = glob.glob(os.path.join(path, '*_AfterCoregistration.npz'))
        assert len(cfiles) == 1
        info['coregnpz'] = cfiles[0]


        outlist.append(info)
        outdict[name] = info

    return outdict



def plot_diff(ax, interferogram1, interferogram2, name1, name2):
    delta_ang =  np.unwrap(interferogram1 - interferogram2, axis=0)
    delta_ang = (delta_ang + np.pi) % (2 * np.pi) - np.pi

    title = name2 #"s {:d}".format(ii+1, combo[0], combo[1])
    ax.imshow(np.degrees(delta_ang), cmap='hsv')
    for im in ax.get_images():
        im.set_clim([-180, 180])
    ax.set_title(title)

    #filename = "cmp_coreg_{:d}.png".format(ii+1)
    #print("Saving plot " + filename)
    #plt.savefig(filename, dpi=150)



def diffimages(interferograms, name0):
    """ Difference all interferograms against one image """

    nplots = min(999, len(interferograms) - 1)
    ncols = 7
    nrows = int(np.ceil(nplots / ncols))


    fig_combined, ax = plt.subplots(nrows, ncols)
    plotidx = 0
    for name in sorted(interferograms.keys()):
        if name == name0:
            continue
        r, c = plotidx // ncols, plotidx % ncols # TODO: use rael
        plot_diff(ax[r, c], interferograms[name0], interferograms[name], name0, name)
        plotidx += 1
    return fig_combined

def diffshift(coregdata, datadict, name0, methods=[0, 1, 2]):
    """ Difference coregistration shifts against each other """

    #nplots = min(999, len(interferograms) - 1)
    #ncols = 7
    #nrows = int(np.ceil(nplots / ncols))

    ifvals = [2, 4, 10, 40, 80]
    # make one figure per method. multiple ifactors per plot
    #fig1, axs = plt.subplots(2, 1)
    figsize=(8, 10)
    fig1 = plt.figure(figsize=figsize)
    ax_all = fig1.gca() # = plt.subplots(nrows, ncols)
    fig4 = plt.figure(figsize=figsize) # bar plot of average/max
    ax_bar = fig4.gca()

    # Plot signal energy
    #ax_all = axs[0]

    rangey = [float('nan'), float('nan')]
    qkey, qlabel = 'qual', 'max / mean'
    #qkey, qlabel = 'qual2', 'max / size'
    rhoplots = []
    rhofigs = []
    coregdiffplots = []
    for ii, method in enumerate(methods):
        # One figure per method, two plots, coregistration quality and offset comparison
        fig2, axs2 = plt.subplots(4, 1, figsize=figsize)
        # coregistration plots
        fig3, axs3 = plt.subplots(len(ifvals), 1, sharex=True, sharey=True, figsize=figsize)
        rhofigs.append([fig2, fig3])
        # Plot coregistration quality per method
        ax1 = axs2[0]
        for name, info in sorted(datadict.items()):
            #if (2 == info['method'] == method and info['ifactor'] == 10) or \
            #   (info['method'] == method and info['ifactor'] == 4):
            if (info['method'] == method and info['ifactor'] in ifvals):
                xmax = len(coregdata[name][qkey])
                x = np.arange(0, xmax)
                # Not sure if we should divide by ifactor
                ifactor = info['ifactor']
                #ifactor = 1
                y = coregdata[name][qkey]# / ifactor * coregdata[name]['qual']
                ax1.plot(x, y, label="method {:d}, ifactor {:d}".format(info['method'], info['ifactor']), linewidth=1, alpha=0.8)
                ax_all.plot(x, y, label="method {:d}, ifactor {:d}".format(info['method'], info['ifactor']), linewidth=1, alpha=0.8)
                jj = ifvals.index(info['ifactor'])
                rho = coregdata[name]['qual2'].T
                rho = (rho - np.min(rho)) 
                # left right bottom top
                print("rho.shape {:s}: {:s}".format(name, str(rho.shape)))
                extent = (0, xmax, 0, rho.shape[0] / ifactor)
                # disable for speed
                axs3[jj].imshow(rho, aspect='auto', extent=extent)
                #y = coregdata[name]['shift_array']*ifactor + len(rho) / 2
                y = coregdata[name]['shift_array'] + rho.shape[0] / ifactor / 2
                axs3[jj].plot(x, y, color='r', linewidth=1.0, alpha=0.5)
                axs3[jj].set_title("Method={:d} ifactor={:d}".format(method, ifactor))
                rangey[0] = np.nanmin([rangey[0], np.nanmin(y)])
                rangey[1] = np.nanmax([rangey[1], np.nanmax(y)])
                rhoplots.append(axs3[jj])

        ax1.legend(loc='right')
        ax1.grid(True)
        ax1.set_title('Coregistration Quality')
        ax1.set_ylabel(qlabel)


        #ax1 = axs[ii+1]
        ax1 = axs2[1]
        plotnames = []
        for name, info in datadict.items():
            if info['method'] == method and info['ifactor'] in ifvals:
                plotnames.append(name)
        offsets = [] # average offset
        xlabels = []
        for name in sorted(plotnames, reverse=False):
            x = np.arange(0, len(coregdata[name]['shift_array']))
            shiftdelta = coregdata[name]['shift_array'] - coregdata[name0]['shift_array']
            # condition for plotting on semilogy
            shiftdelta = np.abs(shiftdelta) + 1e-9
            ax1.semilogy(x, shiftdelta, label=name[5:], marker='.', markersize=1.25, linewidth=0.0)
            axs2[2].plot(x, shiftdelta, label=name[5:], marker='.', markersize=1.25, linewidth=0.0)

            shiftdelta = np.nanmedian(np.abs(coregdata[name]['shift_array'] - coregdata[name0]['shift_array']))
            offsets.append(shiftdelta)
            xlabels.append(name[5:])

        x = range(len(offsets))
        axs2[3].bar(x, offsets)
        axs2[3].set_xticks(x)
        axs2[3].set_xticklabels(xlabels)
        axs2[3].grid(True)
        axs2[3].set_ylabel('Med. diff (samples)')
        coregdiffplots.append(axs2[3])

        # TODO: plot match quality?

        axs2[2].set_ylabel('Coreg Diff (Samples)')
        ax1.set_title("Method {:d} vs {:s}".format(method, name0))
        ax1.legend(loc="right")
        ax1.grid(True)
        ax1.set_ylabel('Coreg Diff (Samples)')
        ax1.set_ylim(bottom=1e-4)

    rangey = [np.floor(rangey[0]), np.ceil(rangey[1])]
    urange = np.median(y)
    dy = 2.0
    rangey = [urange - dy, urange + dy]


    #####################################
    # Save unzoomed
    for ax in rhoplots:
        pass #ax.set_ylim(*rangey)
        # limit x range to first part for checking)
        #ax.set_xlim([5, 2500])
    for ii, (fig2, fig3) in enumerate(rhofigs):
        name = "rho_method{:d}.png".format(ii)
        fig3.savefig(name, dpi=300)
        logging.info("Saved " + name)
    #####################################



    logging.info("Setting y limits to " + str(rangey))
    for ax in rhoplots:
        ax.set_ylim(*rangey)
        # limit x range to first part for checking)
        ax.set_xlim([5, 2500])

    maxylim = max([ax.get_ylim()[1] for ax in coregdiffplots])

    for ax in coregdiffplots:
        ax.set_ylim([0, maxylim])

    for ii, (fig2, fig3) in enumerate(rhofigs):
        name = "rho_method{:d}_zoom.png".format(ii)
        fig3.savefig(name, dpi=300)
        name = "stats_method{:d}.png".format(ii)
        fig2.savefig(name, dpi=300)
        logging.info("Saved " + name)



    ax1 = ax_all
    ax1.legend(loc='right')
    ax1.grid(True)
    ax1.set_title('Coregistration Quality')
    ax1.set_ylabel('max / mean')


    #plt.tight_layout()
    return fig1

def compare_all(interferograms, names):

    ncombo = math.factorial(len(interferograms)) // 2
    fig_combined, ax = plt.subplots(1, ncombo)
    for ii, combo in enumerate(itertools.combinations( range(len(interferograms)), 2)):
        print("{:2d}: Comparing {:s} and {:s}".format(ii+1, names[combo[0]], names[combo[1]]))

        fig = plt.figure()

        delta_ang =  np.unwrap(interferograms[combo[0]] - interferograms[combo[1]], axis=0)
        delta_ang = (delta_ang + np.pi) % (2 * np.pi) - np.pi

        plt.imshow(np.degrees(delta_ang), cmap='hsv')
        plt.clim([-180, 180])
        #plt.subplot(313); plt.imshow(np.rad2deg(int_image), aspect='auto', cmap='hsv');
        plt.colorbar()
        #plt.clabel('Radians')
        title = "{:2d} (Degrees)\n{:s}\nvs {:s}".format(ii+1, names[combo[0]], names[combo[1]])
        plt.title(title)

        title = "[{:2d}] Method {:d} vs {:d}".format(ii+1, combo[0], combo[1])
        ax[ii].imshow(np.degrees(delta_ang), cmap='hsv')
        for im in ax[ii].get_images():
            im.set_clim([-180, 180])
        ax[ii].set_title(title)

        filename = "cmp_coreg_{:d}.png".format(ii+1)
        print("Saving plot " + filename)
        plt.savefig(filename, dpi=150)

    fig_combined.savefig('cmp_coreg_all.png')



def main():

    logging.basicConfig(level=logging.INFO)

    #subdirs = ('coregmethod0', 'coregmethod1', 'coregmethod2')
    datadict = organize_subdirs('.')

    #subdirs = [x['path'] for x in datalist]

    # Load interferogram and shift array images
    interferograms = []
    interferograms_all = {}
    coregdata = {}
    shift_arrs = {}
    names = []
    for name in datadict.keys():
        info = datadict[name]
        npzfile = os.path.join(info['path'], 'ri_int_image_noroll.npz')
        #npzs = [os.path.join(sdir, 'ri_int_image_noroll.npz') for sdir in subdirs]

        print("Loading " + npzfile)
        with np.load(npzfile) as data:
            if info['ifactor'] == 10:
                interferograms.append(data['int_image_noroll'])
                names.append(info['path'])

            #if info['method'] == 0:
            interferograms_all[info['name']] = data['int_image_noroll']

        print("Loading " + info['coregnpz'])
        with np.load(info['coregnpz']) as data:
            loaded_data = {}
            for k in ('shift_array', 'qual', 'qual2'):
                loaded_data[k] = data[k]
            shift_arrs[info['name']] = loaded_data


    fig_combined = diffimages(interferograms_all, 'coregmethod0_if80')
    fig_shift = diffshift(shift_arrs, datadict, 'coregmethod0_if80', methods=range(8))
    #fig_shift = diffshift(shift_arrs, datadict, 'coregmethod0_if10', methods=[0, 1, 2, 3])
    #return
    #print("Doing remaining interferograms ")
    #compare_all(interferograms, names)

    

    plt.show()



if __name__ == "__main__":
    main()
