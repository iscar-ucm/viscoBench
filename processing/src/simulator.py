import os
import yaml
import numpy as np
import pandas as pd

from datetime import datetime
from tqdm import tqdm
from scipy.integrate import solve_ivp

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import src.input_signals as insig


class calc_water_viscosity:
    """
    Calculates the viscosity at some temperature, in Kelvin.

    This is a functor, so it must first be declared (changing the parameters if
    needed), and, after that, it can be called as any other function.
    """
    def __init__(self, A=2.414e-5, B=247.8, C=140):
        self.A = A
        self.B = B
        self.C = C
    def __call__(self, temp_K):
        return self.A * 10 ** (self.B / (temp_K - self.C))


class SystemSimulator:
    def __init__(self) -> None:
        self.ranges_dict = {}

    def load_values(self, config_path):
        # circuit_elements.yaml
        with open(config_path) as cfg_f:
            elements = yaml.safe_load(cfg_f)["elements"]
        # Use the above dictionaries to define the ranges
        self.ranges_dict = {}
        for ky, element in elements.items():
            self.ranges_dict[ky] = {}
            for param in set(element.keys())-set(["e_sources", "e"]):
                if param in element["e_sources"]:
                    self.ranges_dict[ky][param] = \
                        (element[param] - element["e"][param+"_e"],
                         element[param] + element["e"][param+"_e"])
                else:
                    self.ranges_dict[ky][param] = (element[param], element[param])
        print("Simulating with ranges:")
        print(self.ranges_dict)
    
    def load_conditions(self, cfg_path):
        with open(cfg_path) as cfg_f:
            cfg = yaml.safe_load(cfg_f)
        self.t_sim = cfg["t_sim"]
        self.segments_range = cfg["segments_range"]
        self.sliding_points = cfg["sliding_points"]
        self.t_points = cfg["t_points"]
        self.N = cfg["N"]
        self.rand_visc = cfg["rand_visc"]
        self.inc_t = cfg["inc_t"] # Include time in X
        self.visc_range = cfg["visc_range"]
        self.temp_C = cfg["temp_C"]
        self.A_range = cfg["A_range"]
        self.offset_range = cfg["offset_range"]
        self.k_range = cfg["k_range"]
        self.sim_conditions = cfg["sim_conditions"]
    
    def _save_data(self, X, Y, incT, metadata, output_path):
        today = datetime.now()
        self.datafolder = output_path + today.strftime("/dataset_%Y_%m_%d_%H_%M")
        
        if not os.path.exists(output_path):
            os.mkdir( output_path )
        if not os.path.exists(self.datafolder):
            os.mkdir( self.datafolder )

        # Save as numpy binary files
        np.save(self.datafolder+"/X_samples.npy", X)
        np.save(self.datafolder+"/Y_samples.npy", Y)
        np.save(self.datafolder+"/incT_samples.npy", incT)
        np.save(self.datafolder+"/metadata.npy", metadata)

    def _save_format_cfg(self, X, Y, output_path):
        """
        It is assumed that self._save_data() has been called before so that
        the folders are already created.
        """
        X = np.array(X)
        Y = np.array(Y)

        norm_params = {}
        norm_params["in"] = {}
        norm_params["in"]["min"] = [np.min(X)]
        norm_params["in"]["max"] = [np.max(X)]
        norm_params["out"] = {}
        norm_params["out"]["min"] = [np.min(np.transpose(Y)[0])]
        norm_params["out"]["max"] = [np.max(np.transpose(Y)[0])]

        with open(self.datafolder+"/model_cfg.yaml", 'w') as yaml_file:
            yaml.dump(norm_params, stream=yaml_file, default_flow_style=None)

    def _calc_cil_resistance(self, D, L, visc):
        """
        Calculates the hydraulic resistance for a cylindrical tube,
        for a given viscosity.
        """
        return 8*visc*L / (np.pi*((D/2)**4))

    def _calc_cil_capacitance(self, D, L, beta, E, t):
        """
        Calculates the theoretical capacitance of a cylindrical tube.
        """
        # Beta := Bulk Modulus of Water
        V0 = np.pi*(D/2)**2 * L
        return (V0 / beta)*( 1 + ((beta*D)/(E*t)) )

    def _circuit_dyn_eqs(self, t, dP, args): # R0, R1, R2, C1, C2, Pin
        """
        Calcualtes the differential values of pressure drop over the
        two RC branches of the circuit.
        """
        P1, P2 = dP # Integration of previous dP1, dP2
        Rp = args["R0"]
        R1 = args["R1"]
        R2 = args["R2"]
        Cp = args["C1"]
        C2 = args["C2"]
        Pin = args["Pin"](t)

        dP1 = ((Pin - P1)/(Rp * Cp)) - ((P1 - P2) / (R1 * Cp))
        dP2 = ((P1 - P2)/(R1 * C2))- ((P2) / (R2 * C2))
        return [dP1, dP2]
    
    def generate_data(self, output_path):
        """
        Simulate the system under all the required conditions with random
        values within the defined ranges.
        """
        X, Y, incT, metadata = [], [], [], []

        for sim_i, sim_cond in enumerate(self.sim_conditions):
            print("Simulating with conditions set {} / {}".format(
                sim_i+1, len(self.sim_conditions)))
            for n in tqdm(range(self.N)): # For progress bar
                # Get the values of the parameters
                A = np.random.uniform(self.A_range[0], self.A_range[1])
                # Control offset range based on A s.t. values remain below 1.0
                offset = np.random.uniform(self.offset_range[0], 
                                           self.offset_range[1])
                # Reset it if there is no space for offsets
                if offset > (1.0 - A): offset = 0.0
                k = np.random.uniform(self.k_range[0], self.k_range[1])
                sim_params = {
                    "Pin": getattr(insig, sim_cond["sig"])(A=A, k=k, offset=offset)
                }
                # Get value of viscosity within range
                if self.rand_visc:
                    visc = np.max([0.1e-4, np.random.uniform(*self.visc_range)])
                else:
                    visc_calc = calc_water_viscosity()
                    visc = visc_calc(self.temp_C+273.15)

                for ky, element_ranges in self.ranges_dict.items():
                    # Extract the random values for the parameters
                    args = {}
                    for param, param_ranges in element_ranges.items():
                        # We just take the bigger element. No random
                        args[param] = np.max([0.0, np.random.uniform(*param_ranges)])
                    # Resistance
                    if ky[0] == "R":
                        sim_params[ky.split("_")[0]] = \
                            self._calc_cil_resistance(**args, visc=visc)
                    # Capacitance
                    elif ky[0] == "C":
                        sim_params[ky.split("_")[0]] = \
                            self._calc_cil_capacitance(**args)
                    else: 
                        print("Error: element {} not recognized".format(ky))
                        exit()
                    
                # Perform the simulation
                sol = solve_ivp(self._circuit_dyn_eqs, [0, self.t_sim], 
                                [0, 0], args=(sim_params,), dense_output=True)
                
                # Save the results
                # Get the points from the simulation for all segments
                n_segments = np.random.uniform(*self.segments_range)
                t = np.linspace(0, self.t_sim, int(self.t_points*n_segments))
                # n_segments is now only used to determine time length of window
                P1_t = sol.sol(t)[0]
                P2_t = sol.sol(t)[1]

                # Value independent of RgC and sampling time
                R1 = sim_params["R1"]
                R2 = sim_params["R2"]
                # tau = (R1+R2)*sim_params["C2"]
                # tau = R2*sim_params["C2"]
                tau = (R1*R2*sim_params["C2"])/(R1+R2)
                a = (t[1]-t[0]) / tau

                # The results are divided in segments_cnt segments
                for idx in range(0, len(t), self.sliding_points):
                    if (idx+self.t_points) > len(t):
                        # Don't overflow
                        break
                    # t_points in order of 250, 500, 1000, 1500 ...
                    # sliding_points in order of 25, 50, 75, ...
                    P1_t_i = np.array(P1_t[idx:idx+self.t_points])
                    P2_t_i = np.array(P2_t[idx:idx+self.t_points])
                    if self.inc_t:
                        t_i = np.array(t[idx:idx+self.t_points]) - t[idx]
                    # Add measurement noise
                    if sim_cond["noise"] > 0.0:
                        noise_var = np.random.uniform(0.0, sim_cond["noise"])
                        P1_t_i += np.random.normal(0.0, noise_var, np.shape(P1_t_i))
                        P2_t_i += np.random.normal(0.0, noise_var, np.shape(P2_t_i))

                    # If you want to print some curves, don't make N very large
                    # plt.plot(t[idx*self.t_points:(idx+1)*self.t_points], P2_t_i)
                    # plt.plot(t[idx*self.t_points:(idx+1)*self.t_points], P1_t_i)
                    # plt.show()

                    # All required data to recover viscosity from "a" is stored as well
                    if self.inc_t:
                        X.append( np.transpose([P1_t_i, P2_t_i, t_i]) )
                        Y.append( [a, tau, visc, R1, R2, sim_params["C2"], (t[1]-t[0])] )
                    else: 
                        X.append( np.transpose([P1_t_i, P2_t_i]) )
                        Y.append( [a, tau, visc, R1, R2, sim_params["C2"], (t[1]-t[0])] )
                    incT.append( t[1]-t[0] )
                    metadata.append( [sim_cond["sig"], sim_cond["noise"]] )

        # Save the data itself
        self._save_data(X, Y, incT, metadata, output_path)

        # Save the values for normalization
        self._save_format_cfg(X, Y, output_path)

        print("Done.")