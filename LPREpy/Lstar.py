import numpy as np
import logging as log
log.basicConfig(level=log.INFO)
#  Characteristic length calculated using Spalging relations


class Lstar:
    def __init__(self, cea_parameters: list, prop_parameters: list,
                 injector_parameters: list, engine_parameters: list, L_star: float = None):
        # Universal Constants
        self.R_uni = 8.314  # J/mol K   # universal constant of the perfect gases

        # Engine Parameters
        self.Pc = engine_parameters[0]  # chamber pressure, Pa
        self.dc = engine_parameters[1]
        self.Ac = (np.pi / 4) * self.dc ** 2  # cross-sectional area of the combustion chamber
        self.dt = engine_parameters[2]
        self.At = (np.pi / 4) * self.dt ** 2  # cross-sectional area of the throat
        self.m_dot = engine_parameters[3]  # mass flow in the combustion chamber

        # Injector Parameters [do, v_inj, Pc, type, phase_condition]
        self.do = injector_parameters[0]  # diameter injector
        self.v_inj = injector_parameters[1]  # injection velocity of injector
        self.injector_type = injector_parameters[2]  # jet, swirl, pintle
        self.phase_cond = injector_parameters[3]  # l, g, h

        # CEA PARAMETERS
        self.gamma_c = cea_parameters[0]  # gamma in the combustion chamber
        self.v_t = cea_parameters[1]  # gas velocity at the throat
        self.cp_c = cea_parameters[2]  # specific heat at constant pressure of the combustion chamber
        self.T_c = cea_parameters[3]  # T_g, gas temperature
        self.k_c = cea_parameters[4]  # thermal conductivity of combustion chamber
        self.pho_c = cea_parameters[5]  # gas specific mass of combustion chamber
        self.Pr = cea_parameters[6]  # Prandtl number in the combustion chamber

        # NEW PARAMETERS
        self.pho_prop = prop_parameters[0]  # density
        self.st_prop = prop_parameters[1]  # superficial tension
        self.vis_cin_prop = prop_parameters[2]  # cinematic viscosity                 #
        self.Qb_prop = abs(prop_parameters[3])  # heat (enthalpy) of formation
        self.Cv_prop = prop_parameters[4]  # calorific power
        self.T_s_prop = prop_parameters[5]  # boiling temperature
        self.MM_prop = prop_parameters[6]  # molecular mass
        self.wox = prop_parameters[7]  # oxygen concentration
        self.rox = prop_parameters[8]  # mass oxygen of the mixture

        self.R = self.R_uni / self.MM_prop  # gas constant

        # to be calculated
        self.ro = None  # initial radius of the drop
        self.smd = None  # sauter mean diameter
        self.Lstar = L_star  # Characteristic length
        self.ts = None  # tempo de residencia

    def Dp(self, cond='L'):
        cond = cond.lower()
        if cond == 'l':
            return 80 * np.sqrt(10 * self.Pc)
        elif cond == 'g':
            return 40 * np.sqrt(10 * self.Pc)
        else:
            log.error("LPREpy: Characteristic length: phase must be 'l' (liquid) or 'g' (gaseous). ")
            return

    def SMD_function(self):
        if self.injector_type == 'jet':
            self.smd = (500 * (self.do ** 1.2) * (self.vis_cin_prop ** 0.2)) / self.v_inj
            return self.smd
        elif self.injector_type == 'swirl':
            dp = self.Dp(cond=self.phase_cond)
            self.smd = 7.3 * (self.st_prop ** 0.6) * (self.vis_cin_prop ** 0.2) * (self.m_dot ** 0.25) * (dp ** (-0.4))
            return self.smd

    # Lstar parameters
    def Bt(self):
        termo = self.cp_c * ((self.T_c - self.T_s_prop) / self.Qb_prop)
        return self.cp_c * ((self.T_c - self.T_s_prop) / self.Qb_prop)

    def B(self):
        termo = (self.Cv_prop * self.wox) / (self.rox * self.Qb_prop)
        return termo + self.Bt()

    def Sr(self):
        return (9 * self.Pr) / (np.log(1 + self.B()))

    def Xo(self):
        return self.v_inj / self.v_t

    def E_star(self):
        sr = self.Sr()
        xo = self.Xo()
        return (xo + 0.3 * sr) / (2 + sr)

    # Lstar calculation
    def L_star(self):
        if isinstance(self.Lstar, float):
            return self.Lstar

        ro = self.SMD_function() / 2

        g_c = self.m_dot / self.Ac

        termo1_gama = (self.gamma_c - 1) / (self.gamma_c + 1)

        termo2_gama = (self.gamma_c + 1) / (2 * self.gamma_c - 2)

        termo3_gama = 2 / (self.gamma_c + 1)

        termo1 = self.E_star() * (ro ** 2)

        termo2 = g_c / (self.pho_c * np.sqrt(self.gamma_c * self.R * self.T_c))

        termo3 = ((self.cp_c * self.pho_prop) / self.k_c)

        termo4 = np.sqrt(self.gamma_c * self.R * self.T_c) / np.log(1 + self.B())

        self.Lstar = termo1 * ((termo3_gama + (termo2 ** 2) * termo1_gama) ** termo2_gama) * termo3 * termo4

        return self.Lstar

    def T_s(self):
        exp = (self.gamma_c + 1) / (2 * self.gamma_c - 2)
        Ec = self.Ac/self.At
        termo1 = (Ec*self.Lstar)/np.sqrt(self.gamma_c*self.R*self.T_c)
        termo2 = (self.gamma_c+1)/2
        self.ts = termo1 * (termo2 ** exp)
        return self.ts  # s
