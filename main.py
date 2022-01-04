import os.path

import matplotlib.pyplot as plt
import toml

import roads.curve_find as curve_find
from roads.curve_fit import fitCurves2
from roads.curves import saveCurves, loadCurves, saveReport, printCurves
from roads.svgread import loadSvg


# load config from file into parameters
def load_config(config_file_name, parameters=None):
    if not os.path.exists(config_file_name):
        print(f"No config file {config_file_name} found, using defaults")
        return

    with open(config_file_name, 'rt') as fi:
        data = toml.load(fi)

    if not parameters:
        parameters = {}

    if 'finding' not in parameters:
        parameters['finding'] = {}
    if 'fitting' not in parameters:
        parameters['fitting'] = {}

    try:
        parameters['finding'].update(data['finding'])
    except:
        pass
    try:
        parameters['fitting'].update(data['fitting'])
    except:
        pass

    if 'sequence' in data:
        try:
            parameters['sequence'].update(data['sequence'])
        except:
            parameters['sequence'] = data['sequence']

    return parameters


def main(filename, parameters, try_load=True, plot_curvature=False, show=False):
    curves_file_name = filename.replace('.svg', '-curves.json')
    report_file_name = filename.replace('.svg', '-report.csv')
    config_file_name = filename.replace('.svg', '.cfg')
    image_file_name = filename.replace('.svg', '-plots.png')
    curvature_image_name = filename.replace('.svg', '-curvature.png')

    path = loadSvg(filename)

    load_config(config_file_name, parameters)

    path.makeContinuousCurvature()

    print(parameters)
    fig1, ax1 = plt.subplots(2 if plot_curvature else 1, 1)
    fig2, ax2 = plt.subplots(2 if plot_curvature else 1, 1)

    # if not plot_curvature:
    ax = [ax1, ax2]

    # ax[0].invert_yaxis()
    ax[0].axis('equal')

    path.plot('black', sampler_fn="position", show=False, axis=ax[0])

    # if plot_curvature:
    path.plot("r:", sampler_fn="curvature", show=False, axis=ax[1])

    curves = None
    loaded = False
    if try_load:
        try:
            curves = loadCurves(path, curves_file_name)
            loaded = True
        except:
            pass

    if "sequence" in parameters:
        pattern = parameters['sequence']['pattern']
        print("Using pattern: ", pattern)
        curves = curve_find.fromPattern(path, pattern, parameters['finding'])
    else:
        curves = curve_find.buildCurves(path, parameters['finding'])

    print("Before fitting")
    printCurves(curves)

    if not loaded:
        fitCurves2(curves, parameters['fitting'])
        if "sequence" not in parameters:
            curve_find.simplifyCurves(curves, 1, 0.5, 1)
            fitCurves2(curves, parameters['fitting'])

    # if not loaded:
    # growArcs(curves)

    saveCurves(curves, curves_file_name)
    saveReport(curves, report_file_name, path)

    print("After fitting")
    printCurves(curves)

    tot_len = 0
    for i, c in enumerate(curves):
        c.plot("-", axis=ax[0], label=c.legend)
        tot_len += c.length
    # do not display legend
    # ax[0].plot([], [], ' ', label=f"Total length={tot_len:.2f}m")
    # ax[0].legend()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16,
            }
    ax[0].set_xlabel('Coordinate real value', font)
    ax[0].set_ylabel('Coordinate image value', font)

    # if plot_curvature:
    for i, c in enumerate(curves):
        c.plot(axis=ax[1], plot_curvature=True, color='black')
    # set legend
    ax[1].plot([], [], color='black', linewidth=2.0, label='Linear fit')
    ax[1].plot([], [], color='red', linewidth=2.0, linestyle='--', label='Curvature result')
    ax[1].legend(fontsize=20)
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 20,
             }
    ax[1].set_xlabel('Curvature real value (1/m)', font2)
    ax[1].set_ylabel('Curvature image value (1/m)', font2)

    for c in curves:
        cname = c.__class__.__name__
        err = c.getErrorTally()

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 2, Size[1] * 2,
                      forward=True)  # Set forward to True to resize window along with plot in figure

    fig1.savefig(image_file_name)
    fig2.savefig(curvature_image_name)
    if show:
        plt.show()


if __name__ == '__main__':
    parameters = load_config("default.cfg")

    ramps = [
        "./traced/ramp1/ramp1.svg",
        "./traced/ramp2/ramp2.svg",
        "./traced/ramp3/ramp3.svg",
        "./traced/ramp4/ramp4.svg",
        "./traced/ramp5/ramp5.svg",
        "./traced/ramp6/ramp6.svg",
    ]

    for f in ramps:
        name = os.path.split(f)[1]
        print(f"========== Processing: {name} =========")
        main(f, parameters, try_load=False, plot_curvature=False, show=False)
