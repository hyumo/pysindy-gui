
from dearpygui.core import *
from dearpygui.simple import *

import pandas as pd
import numpy as np
import pysindy as ps
from sklearn.linear_model import Lasso
import functools

import pysindygui.config as cfg


def configure_items(names, **kwargs):
    for name in names:
        configure_item(name, **kwargs)

def file_select_cb(sender, data):
    filename = f'{data[0]}//{data[1]}'
    #filename = "D:/startup/software/pysindygui/data/lorenz.csv"
    #filename = "D:/startup/software/pysindygui/data/firstOrder.csv"
    try: 
        df = pd.read_csv(filename)
    except:
        log_error(f"Failed to load {filename}", logger="logger##main")
        return
    log_info(f"Loaded {filename}", logger="logger##main")
    log_debug(f"columns: {df.columns.to_list()}", logger="logger##main")

    # Check if time channel exists
    if 'Time' in df.columns:
        df.rename(columns={"Time": "time"})
    if 'TIME' in df.columns:
        df.rename(columns={"TIME": "time"})

    # Trying to be smart here.
    # - Add time data to store
    # - Disable step size textbox
    # - Enable the 'time' channel combo and auto select 'time'
    if 'time' in df.columns:
        log_debug("Found \'time\' column in csv")
        add_data(cfg.TIME, df['time'].to_numpy())
        set_value("Method##time", "From Channels")
        configure_item("Step size##time", enabled=False)
        configure_item("Channel##combo##time", enabled=True)
        set_value("Channel##combo##time", "time")

    channel_names = df.columns.to_list()
    # Configure channels item
    channel_items = ["Channels##channels", "Channel##combo##time",
                     "Channels##combo##states", "Channels##combo##der", "Channels##combo##inputs"]
    configure_items(channel_items, items=channel_names)
    configure_item("Fit model##fitting", enabled=True)

    # add_data to store
    add_data(cfg.CSV, df)
    add_data(cfg.CHANNEL_NAMES, channel_names)
    add_data(cfg.X_NAMES, [])
    add_data(cfg.U_NAMES, [])
    add_data(cfg.DER_NAMES, [])


def fit_model_cb(sender, data):
    df = get_data(cfg.CSV)
    
    # Get x data
    x_names = get_data(cfg.X_NAMES)
    X = df[x_names].to_numpy()
    add_data(cfg.X, X)
    
    # Get der data

    # Get input data
    u_names = get_data(cfg.U_NAMES)
    if len(u_names) > 0:
        U = df[u_names].to_numpy()
    else:
        U = None

    add_data(cfg.U, U)

    # Get time data
    if get_value("Method##time") == "Constant Step":
        dt = get_value("Step size##time")
        time = np.arange(0, X.shape[0])*dt
    else:
        time_channel = get_value("Channel##combo##time")
        if time_channel == None or time_channel == "":
            log_error("Time channel must be selected", logger="logger##main")
            return
        time = df[time_channel].to_numpy()
    add_data(cfg.TIME, time)

    # Get Optimizer
    if get_value("Methods##Optimizers") == "STLSQ":
        threshold = get_value("threshold##optimizers")
        alpha = get_value("alpha##optimizers")
        max_iter = get_value("max_iter##optimizers")
        optimizer = ps.STLSQ(threshold=threshold,
                             alpha=alpha, max_iter=max_iter)
    elif get_value("Methods##Optimizers") == "Lasso":
        alpha = get_value("alpha##optimizers")
        max_iter = get_value("max_iter##optimizers")
        optimizer = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=False)
    else:
        optimizer = None

    ##############################
    # Feature libraries
    ##############################
    libs = []
    if get_value("Enable##polynomial##libraries") == True:
        degree = get_value("Degree##polynomial##libraries")
        include_interaction = not (
            get_value("Type##polynomial##libraries") == "Only states")
        interaction_only = get_value(
            "Type##polynomial##libraries") == "Only interaction"
        include_bias = get_value(
            "Include bias terms##polynomial##libraries")
        log_debug(
            f'degree: {degree}, include_interaction: {include_interaction}, interaction_only: {interaction_only}, include_bias: {include_bias}')
        libs.append(ps.PolynomialLibrary(degree=degree, include_interaction=include_interaction,
                                         interaction_only=interaction_only, include_bias=include_bias))

    if get_value("Enable##fourier##libraries") == True:
        n_frequencies = get_value("n_frequencies##fourier##libraries")
        include_sin = get_value("Include sin##fourier##libraries")
        include_cos = get_value("Include cos##fourier##libraries")
        try:
            fourierlib = ps.FourierLibrary(
                n_frequencies=n_frequencies, include_sin=include_sin, include_cos=include_cos)
            libs.append(fourierlib)
        except ValueError as err:
            log_error(err, "logger##main")
            return

    if get_value("Enable##identity##libraries") == True:
        libs.append(ps.IdentityLibrary())

    # Handle the case if nothing's selected
    if not libs:
        libs.append(ps.PolynomialLibrary())
    
    log_debug(libs, logger="logger##main")

    # Get "feature_library" by reducing the "libs" list
    feature_library = functools.reduce(lambda a, b: a+b, libs)
    try:
        model = ps.SINDy(optimizer=optimizer, feature_library=feature_library)
        model.fit(X, t=time, u=U)
        log_info(f"Model fitted.", logger="logger##main")
    except ValueError as err:
        log_error(err, logger="logger##main")
        return

    model_eqs = []
    for i, eq in enumerate(model.equations()):
        model_eqs.append(f"der(x{i}) = {eq}")
    model_text = "\n".join(model_eqs)

    set_value("Score##fitting", model.score(X, time, u=U))
    set_value("Equations##fitting", model_text)
    add_data(cfg.MODEL, model)

    # clear X_fit listbox
    configure_item("X_fit##fitting", items=[])


def optimizers_section():
    ''' Optimizer section'''
    with collapsing_header("Optimizers##main", default_open=True):
        def methods_select_cb(sender, data):
            if get_value(sender) == "STLSQ":
                configure_item("threshold##optimizers", enabled=True)
                set_value("threshold##optimizers", 0.1)
                set_value("alpha##optimizers", 0.05)
                set_value("max_iter##optimizers", 20)
            else:
                configure_item("threshold##optimizers", enabled=False)
                set_value("alpha##optimizers", 0.1)
                set_value("max_iter##optimizers", 1000)

        add_combo("Methods##Optimizers", items=[
            "STLSQ", "Lasso"], default_value="STLSQ", callback=methods_select_cb)

        add_input_float("threshold##optimizers", default_value=0.1,
                        tip="Minimum magnitude for a coefficient in the weight vector. Coefficients with magnitude below the threshold are set to zero")
        add_input_float("alpha##optimizers", default_value=0.05,
                        tip="L1 (Lasso) or L2 (STLSQ) regularization on the weight vector.")
        add_input_int("max_iter##optimizers", default_value=20,
                      tip="Maximum iterations of the optimization algorithm")


def feature_libraries_section():
    '''Feature libraries section'''
    with collapsing_header("Libraries##main", default_open=True):
        with tab_bar("Libraries##tabbar##libraries"):
            # Polynomial tab
            with tab("Polynomial##tab##libraries"):
                add_checkbox("Enable##polynomial##libraries", default_value=True, callback=lambda sender, data: configure_items(
                    ["Degree##polynomial##libraries",
                        "Type##polynomial##libraries", "Include bias terms##polynomial##libraries"], enabled=get_value("Enable##polynomial##libraries")))
                add_input_int("Degree##polynomial##libraries",
                              default_value=2, min_value=0)
                # TODO: It seems that only interaction is buggy
                add_combo("Type##polynomial##libraries", items=[
                    "States and interaction", "Only states"], default_value="States and interaction")
                add_checkbox(
                    "Include bias terms##polynomial##libraries", default_value=True)
            # Fourier tab
            with tab("Fourier##tab##libraries"):
                add_checkbox("Enable##fourier##libraries",
                             default_value=False, callback=lambda sender, data: configure_items(
                                 ["n_frequencies##fourier##libraries",
                                     "Include sin##fourier##libraries", "Include cos##fourier##libraries"], enabled=get_value("Enable##fourier##libraries")))
                add_input_int("n_frequencies##fourier##libraries",
                              min_value=1, default_value=1, enabled=False)
                add_checkbox("Include sin##fourier##libraries",
                             default_value=True, enabled=False)
                add_checkbox("Include cos##fourier##libraries",
                             default_value=True, enabled=False)

            # Identity tab
            with tab("Identity##tab##libraries"):
                add_checkbox("Enable##identity##libraries",
                             default_value=False)


def fitting_section():
    ''' Model fitting section'''
    with collapsing_header("Fitting##main", default_open=True):
        def xfit_select_cb(sender, data):
            index = get_value(sender)
            x_names = get_data(cfg.X_NAMES)
            x_name = x_names[index]
            X_fit = get_data(cfg.X_FIT)
            time = get_data(cfg.TIME)
            add_line_series("figure##plot", f"{x_name}_fit", time[:np.shape(
                X_fit)[0]], X_fit[:, index].ravel())

        def simulate_cb(sender, data):
            X = get_data(cfg.X)
            X0 = X[0, :]
            time = get_data(cfg.TIME)
            u = get_data(cfg.U)
            model = get_data(cfg.MODEL)
            X_fit = model.simulate(X0, time, u=u)
            add_data(cfg.X_FIT, X_fit)
            configure_item("X_fit##fitting",
                           items=get_data(cfg.X_NAMES))
            log_info("Simulate fitted model.", logger="logger##main")

        add_button("Fit model##fitting", height=50, width=-1,
                   callback=fit_model_cb, enabled=False)
        add_input_text("Score##fitting", readonly=True)
        add_input_text("Equations##fitting",
                       multiline=True, readonly=True)

        add_button("Simulate##fitting",
                   tip="Simulate fitted model", callback=simulate_cb)
        add_listbox("X_fit##fitting", items=[],
                    callback=xfit_select_cb)


def data_section():
    with collapsing_header("Load data##main", default_open=True):
        def channel_select_cb(sender, data):
            df = get_data(cfg.CSV)
            channel = df.columns[get_value(sender)]
            time = get_data(cfg.TIME)
            if time is None:
                dt = 1
                count = len(df[channel])
                time = np.arange(0, count)*dt
            # Plot selected channel
            add_line_series(
                "figure##plot", f"{channel}", time, df[channel].to_numpy(dtype=np.float64))

        #add_button("I WANT DATA!", callback=file_select_cb)

        add_listbox("Channels##channels", items=[],
                    callback=channel_select_cb)
        with tab_bar("loaddata##tabbar##main"):
            # Time tab
            with tab("Time##tab"):
                def method_select_cb(sender, data):
                    if get_value(sender) == "Constant Step":
                        configure_item("Step size##time", enabled=True)
                        configure_item(
                            "Channel##combo##time", enabled=False)
                    else:
                        configure_item(
                            "Step size##time", enabled=False)
                        configure_item(
                            "Channel##combo##time", enabled=True)
                add_text("Time: ")
                add_combo("Method##combo##time", items=[
                    "Constant Step", "From Channels"], default_value="Constant Step", callback=method_select_cb)
                add_input_float("Step size##time",
                                default_value=0.1, min_value=0.0, step=0.1)
                add_combo("Channel##combo##time",
                          items=[], enabled=False)

            # States tab
            with tab("States##tab"):
                def state_select_cb(sender, data):
                    if get_value('Channels##combo##states') == None or get_value('Channels##combo##states') == '':
                        return
                    selected = get_value('Channels##combo##states')
                    x_names = get_data(cfg.X_NAMES)
                    if sender == "+##states":
                        if selected not in x_names:
                            x_names.append(selected)
                            add_data(cfg.X_NAMES, x_names)
                    else:
                        if selected in x_names:
                            x_names.remove(selected)
                            add_data(cfg.X_NAMES, x_names)

                    tabledata = []
                    for index, name in enumerate(x_names):
                        row = [f'x{index}', name]
                        tabledata.append(row)
                    set_table_data("Table##states", tabledata)
                add_button("+##states", callback=state_select_cb)
                add_same_line()
                add_button("-##states", callback=state_select_cb)
                add_same_line()
                add_combo("Channels##combo##states", items=[])
                add_table("Table##states", [
                    "Feature name", "State"], height=100)

            # Inputs tab
            with tab("Inputs(O)##tab"):
                def input_select_cb(sender, data):
                    if get_value('Channels##combo##inputs') == None or get_value('Channels##combo##inputs') == '':
                        return
                    selected = get_value('Channels##combo##inputs')
                    u_names = get_data(cfg.U_NAMES)
                    if sender == "+##inputs":
                        if selected not in u_names:
                            u_names.append(selected)
                            add_data(cfg.U_NAMES, u_names)
                    else:
                        if selected in u_names:
                            u_names.remove(selected)
                            add_data(cfg.U_NAMES, u_names)

                    tabledata = []
                    for index, name in enumerate(u_names):
                        row = [f'x{index}', name]
                        tabledata.append(row)
                    set_table_data("Table##inputs", tabledata)

                add_button("+##inputs", callback=input_select_cb)
                add_same_line()
                add_button("-##inputs", callback=input_select_cb)
                add_same_line()
                add_combo("Channels##combo##inputs", items=[])
                add_table("Table##inputs", [
                    "Feature name", "inputs"], height=100)

            # Der tab
            with tab("Derivative(O)##tab", show=False):
                def der_select_cb(sender, data):
                    if get_value('Channels##combo##der') == None or get_value('Channels##combo##der') == '':
                        return
                    selected = get_value('Channels##combo##der')
                    der_names = get_data(cfg.DER_NAMES)
                    if sender == "+##der":
                        if selected not in der_names:
                            der_names.append(selected)
                            add_data(cfg.DER_NAMES, der_names)
                    else:
                        if selected in der_names:
                            der_names.remove(selected)
                            add_data(cfg.DER_NAMES, der_names)

                def enable_items(sender, data):
                    if get_value(sender) == "From Channels":
                        configure_items(
                            ["+##der", "-##der", "Channels##combo##der"], enabled=True)
                    else:
                        configure_items(
                            ["+##der", "-##der", "Channels##combo##der"], enabled=False)

                add_combo("Method##der", items=[
                    "Finite Difference", "Smooth Finite Difference", "From Channels"], default_value="Finite Difference", callback=enable_items)
                add_button("+##der", callback=der_select_cb,
                           enabled=False)
                add_same_line()
                add_button("-##der", callback=der_select_cb,
                           enabled=False)
                add_same_line()
                add_combo("Channels##combo##der",
                          items=[], enabled=False)
                add_table("Table##der", [
                    "Feature name", "Der"], height=100)


def menu_section():
    with menu_bar("MenuBar##main"):
        with menu("Menu##main"):
            add_menu_item("Load csv##main", callback=lambda sender, data: open_file_dialog(
                file_select_cb, ".*,.csv"))
        with menu("Help##main"):
            add_menu_item("About##main", callback=lambda sender, data: configure_item("About", show=True))


class PySINDyGUI(object):
    def __init__(self):
        self.debug = True
        enable_docking(shift_only=False, dock_space=True)

        set_main_window_size(1320, 950)
        set_main_window_title("PySINDy")
        set_main_window_pos(200, 0)
        with window("Main Window", width=400, height=900, x_pos=0, y_pos=0, no_close=True):
            menu_section()
            # Load data
            data_section()
            # Libraries
            feature_libraries_section()
            # Optimizers
            optimizers_section()
            # Fitting
            fitting_section()

        with window("Plots", width=900, height=550, x_pos=400, y_pos=0, no_close=True):
            add_button("Clear plots##fitting", tip="Clear current plots",
                       callback=lambda sender, data: clear_plot("figure##plot"))
            add_plot("figure##plot")

        with window("Logger", width=900, height=350, x_pos=400, y_pos=550, no_close=True):
            add_logger("logger##main", autosize_x=True, autosize_y=True)
        
        with window("About", width=280, height=150, x_pos=500, y_pos=300, show=False):
            add_text("Name: PySINDy-gui v1.0")
            add_separator()
            add_text("Author: Hang Yu (hyumo)")
            add_text("Email: yuhang.neu@gmail.com")
            add_text("https://github.com/hyumo/pysindy-gui")


    def run(self):
        if self.debug:
            pass
            #show_logger()
            #show_demo()
        start_dearpygui()
