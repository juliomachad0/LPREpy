import numpy as np
import logging as log
import pandas as pd
from pandas import DataFrame

log.basicConfig(level=log.INFO)


class ImpingementJetInjectors:

    def __init__(self) -> None:
        # engine parameters
        self.of = None
        self.Pc = None
        self.DeltaPc = None
        self.total_mass_flow = None
        self.ox_total_mass_flow = None
        self.fuel_total_mass_flow = None
        # oxidizer properties
        self.ox_vis = None
        self.ox_st = None
        self.ox_pho = None
        self.ox_vap = None
        self.ox_phase = 'l'
        # fuel properties
        self.fuel_vis = None
        self.fuel_st = None
        self.fuel_pho = None
        self.fuel_vap = None
        self.fuel_phase = 'l'
        # cd and results
        self.theoretical_max_cd = 0.81  # based on experiments - book: atomization - levebrev
        self.results_ox = None
        self.results_fuel = None

    def Prop_Properties(self, Ox_props: list, Fuel_props: list):
        log.info("LPREpy: Prop. properties: format of list: [vis, st, pho, vap, phase]")
        log.info(
            "LPREpy: Prop. properties: 'phase' parameters must be string->\n"
            "'l'-liquid, 'g'-gaseous, 'h'-hybrid (arithmetic mean of l and g).\n"
            "Other items: float or None type if unknown. All parameters in S.I (m, kg, s, N, Pa)")
        log.info(
            "LPREpy: Prop. properties: vis-viscosity, st-superficial tension, pho-density, vap-vapor p., phase-state")
        # ox properties
        self.ox_vis = Ox_props[0]
        self.ox_st = Ox_props[1]
        self.ox_pho = Ox_props[2]
        self.ox_vap = Ox_props[3]
        if Ox_props[4] is None:  # if phase is None
            pass
        else:
            self.ox_phase = Ox_props[4]
        # fuel properties
        self.fuel_vis = Fuel_props[0]
        self.fuel_st = Fuel_props[1]
        self.fuel_pho = Fuel_props[2]
        self.fuel_vap = Fuel_props[3]
        if Fuel_props[4] is None:  # if phase is None
            pass
        else:
            self.fuel_phase = Fuel_props[4]

    def Engine_CD_Parameters(self, Pc, m_dot_total, of, DeltaPc=None):
        if DeltaPc is not None or not isinstance(DeltaPc, str):
            self.DeltaPc = DeltaPc
        self.Pc = Pc
        self.of = of
        self.total_mass_flow = m_dot_total
        self.ox_total_mass_flow = (of / (1 + of)) * self.total_mass_flow
        self.fuel_total_mass_flow = (1 / (1 + of)) * self.total_mass_flow

    def Impingement_Jet(self, prop_type: str, n_inj: int, lo=None, lo_do_r=None, theoretical_cd_max=None):
        if theoretical_cd_max is not None and not isinstance(theoretical_cd_max, str):
            self.theoretical_max_cd = theoretical_cd_max
        match prop_type:
            case 'ox':
                pho = self.ox_pho
                vis = self.ox_vis
                phase = self.ox_phase
                mass_flow = self.ox_total_mass_flow

            case 'fuel':
                pho = self.fuel_pho
                vis = self.fuel_vis
                phase = self.fuel_phase
                mass_flow = self.fuel_total_mass_flow
            case _:
                log.error("In prop_type parameter put 'ox' for oxidizer or 'fuel' for fuel")
                return 0

        if (self.DeltaPc is None) or isinstance(self.DeltaPc, str):
            DeltaPc = self.DeltaP(self.Pc, phase)
        else:
            DeltaPc = self.DeltaPc

        # inicio, calculo do 'do' baseado em cd teorico maximo baseado em experimentos
        do_theo = self.Do(n_inj, DeltaPc, mass_flow, pho, self.theoretical_max_cd)
        v_theo = self.v_inj(self.theoretical_max_cd, self.Pc, pho)
        Re_theo = self.Reynolds(do_theo, v_theo, pho, vis)
        if lo is None and lo_do_r is not None:
            lo = lo_do_r * do_theo
        elif lo is not None and lo_do_r is None:
            lo_do_r = lo / do_theo
        elif lo is not None and lo_do_r is not None:
            lo = lo_do_r * do_theo
            log.info("LPREpy:Just one parameter ('lo' or 'lo_do_r' (lo/do)) is necessary.")
            log.info("LPREpy:If both parameters are informed ('lo' or 'lo_do_r'), 'lo_do_r' (lo/do) is considered.")
        elif lo is None and lo_do_r is None:
            log.error("LPREpy: A least one parameter ('lo' or 'lo_do_r' (lo/do)) must be informed.")
            return 0

        # CDs based on experiments
        CD_max_c = self.CD_max(lo, do_theo)
        do_max_c = self.Do(n_inj, DeltaPc, mass_flow, pho, CD_max_c)
        v_max_c = self.v_inj(CD_max_c, self.Pc, pho)
        Re_max_c = self.Reynolds(do_max_c, v_max_c, pho, vis)
        lo_do_r_max_c = lo / do_max_c

        CD_max_c_naka = self.CD_max_naka(lo, do_theo)
        do_max_c_naka = self.Do(n_inj, DeltaPc, mass_flow, pho, CD_max_c_naka)
        v_max_c_naka = self.v_inj(CD_max_c_naka, self.Pc, pho)
        Re_max_c_naka = self.Reynolds(do_max_c_naka, v_max_c_naka, pho, vis)
        lo_do_r_max_c_naka = lo / do_max_c_naka

        CD_naka = self.CD_Nakayama(lo, do_theo, Re_theo)  # CD_Nakayma Re 550 to 7000, lo/do 1.5 to 17
        do_naka = self.Do(n_inj, DeltaPc, mass_flow, pho, CD_naka)
        v_naka = self.v_inj(CD_naka, self.Pc, pho)
        Re_naka = self.Reynolds(do_naka, v_naka, pho, vis)
        lo_do_r_naka = lo / do_naka

        CD_asi = self.CD_Asihmin(lo, do_theo, Re_theo)  # CD_Asihmin Re 100 to 150000, lo/do 2 to 5
        do_asi = self.Do(n_inj, DeltaPc, mass_flow, pho, CD_asi)
        v_asi = self.v_inj(CD_asi, self.Pc, pho)
        Re_asi = self.Reynolds(do_asi, v_asi, pho, vis)
        lo_do_r_asi = lo / do_asi

        CD_lich = self.CD_Lichtarowicz(lo, do_theo, Re_theo)  # CD_Lichtarowicz Re 10 to 20000, lo/do 2 to 10
        do_lich = self.Do(n_inj, DeltaPc, mass_flow, pho, CD_lich)
        v_lich = self.v_inj(CD_lich, self.Pc, pho)
        Re_lich = self.Reynolds(do_lich, v_asi, pho, vis)
        lo_do_r_lich = lo / do_lich
        #   RESULTADOS

        results_cd_max = [CD_max_c, do_max_c, v_max_c, Re_max_c, lo_do_r_max_c]
        results_cd_max_naka = [CD_max_c_naka, do_max_c_naka, v_max_c_naka, Re_max_c_naka, lo_do_r_max_c_naka]
        results_theory = [self.theoretical_max_cd, do_theo, v_theo, Re_theo, lo_do_r]
        results_naka = [CD_naka, do_naka, v_naka, Re_naka, lo_do_r_naka]
        results_asi = [CD_asi, do_asi, v_asi, Re_asi, lo_do_r_asi]
        results_lich = [CD_lich, do_lich, v_lich, Re_lich, lo_do_r_lich]

        axis = ['CD', 'do', 'v_inj', 'Re_lo', 'lo/do']
        data = {
            'Parameters': axis,
            'CD_max': results_cd_max,
            'CD_max_naka': results_cd_max_naka,
            'Theoretical': results_theory,
            'Naka': results_naka,
            'Asi': results_asi,
            'Lich': results_lich
        }
        dataframe: DataFrame = pd.DataFrame(data)
        dataframe.set_index('Parameters', append=False, inplace=True)
        match prop_type:
            case 'ox':
                self.results_ox = dataframe
            case 'fuel':
                self.results_fuel = dataframe

    def Get_results(self, prop_type: str, CD_type: str, parameter='all'):
        parameter_lower = parameter.lower()
        CD_type_lower = CD_type.lower()
        # evaluating propellant type
        match prop_type:
            case 'ox':
                dataframe = self.results_ox
            case 'fuel':
                dataframe = self.results_fuel
            case _:
                log.error("In prop_type parameter put 'ox' for oxidizer or 'fuel' for fuel")
                return 'ERROR'
        # evaluating wrong input parameters
        CD_type_v = ['all', 'CD_max', 'CD_max_naka', 'Theoretical', 'Naka', 'Asi', 'Lich', 'Best']
        CD_type_v_lower = ['all', 'cd_max', 'cd_max_naka', 'theoretical', 'naka', 'asi', 'lich', 'best']
        parameters = ['all', 'CD', 'do', 'v_inj', 'Re_lo', 'lo/do']
        parameters_lower = ['all', 'cd', 'do', 'v_inj', 're_lo', 'lo/do']
        if (not isinstance(CD_type, str)) and (not isinstance(parameter, str)):
            log.error("LPREpy: Input parameters must be strings type.".upper())
            return 'Error'
        if CD_type_lower not in CD_type_v_lower:
            log.error("LPREpy: Wrong input in CD_type parameter informed.".upper())
            log.info("LPREpy: The possible CD types are: {}".upper().format(CD_type_v))
            return 'Error'
        elif parameter_lower not in parameters_lower:
            log.error("LPREpy: Wrong name in informed in 'parameter' option.".upper())
            log.info("LPREpy: The possible parameters are: {}".upper().format(parameters))
            return 'Error'
        # evaluating options
        if CD_type_lower == 'all':
            log.info("LPREpy: The data are in format of Pandas DataFrame.".upper())
            return dataframe
        elif CD_type_lower == 'best':
            data_best_cd, best_CD_type, state_of_decision = self.Best_CD(prop_type=prop_type)
            data_best_do = float(data_best_cd.loc['do'])*1000
            log.info("LPREpy: Best CD type for flow without cavitation: {} CD type, Propellant type: {}, Condition: {}".
                     format(best_CD_type, prop_type, state_of_decision))
            log.info("LPREpy: - CD: {:.4f}, do: {:.2f} MM, lo/do: {:.2f}, v_inj: {:.2} m/s, Re: {:.0f}"
                     .format(data_best_cd.loc['CD'], data_best_do,
                             data_best_cd.loc['lo/do'], data_best_cd.loc['v_inj'], data_best_cd.loc['Re_lo']))
            log.info("LPREpy: The data are in format of Pandas DataFrame.".upper())
            return data_best_cd
        elif parameter_lower == 'all':
            log.info("LPREpy: The data are in format of Pandas DataFrame.".upper())
            return dataframe[CD_type]
        else:
            return float(dataframe[CD_type][parameter])

    def Best_CD(self, prop_type):
        match prop_type:
            case 'ox':
                dataframe = self.results_ox
            case 'fuel':
                dataframe = self.results_fuel
            case _:
                log.error("LPREpy: In prop_type parameter put 'ox' for oxidizer or 'fuel' for fuel")
                return 'ERROR'
        Re = dataframe['Theoretical']['Re_lo']
        lo_do_r = dataframe['Theoretical']['lo/do']

        if (1.5 <= lo_do_r <= 17) and (550 <= Re <= 7000):
            return dataframe['Naka'], 'Naka', 'Fully fitted'
        elif (2 <= lo_do_r <= 5) and (100 <= Re <= 150000):
            return dataframe['Asi'], 'Asi', 'Fully fitted'
        elif (2 <= lo_do_r <= 10) and (10 <= Re <= 20000):
            return dataframe['Lich'], 'Lich', 'Fully fitted'
        elif Re > 150000:
            if 2 <= lo_do_r <= 5:
                return dataframe['Asi'], 'Asi', 'Partially fitted'
        elif Re < 10:
            if 2 <= lo_do_r <= 10:
                return dataframe['Lich'], 'Lich', 'Partially fitted'
        else:
            return dataframe['Theoretical'], 'Theoretical', 'Not fitted'

    @staticmethod
    def DeltaP(pressure_chamber, phase='l'):
        match phase:
            case 'l':
                return 80 * np.sqrt(10 * pressure_chamber)
            case 'g':
                return 40 * np.sqrt(10 * pressure_chamber)
            case 'h':
                return (80 * np.sqrt(10 * pressure_chamber) + 40 * np.sqrt(10 * pressure_chamber)) / 2
            case _:
                log.error(
                    "LPREpy: In the 'phase' parameter -> 'l' for liquid prop., 'g' for gaseous prop. or "
                    "'h' for biphasic prop.")
                log.info("LPREpy: For the 'h' option, the function calculates the arithmetic mean "
                         "between 'l' and 'g' phase.")
                return 0

    @staticmethod
    def Do(n_inj, DPc, m_dot, pho, cd):
        Ao = (m_dot / n_inj) / (cd * np.sqrt(2 * DPc * pho))
        return np.sqrt((4 * Ao) / np.pi)

    @staticmethod
    def v_inj(CD, DeltaPc, pho):
        return CD * np.sqrt((2 * DeltaPc) / pho)

    @staticmethod
    def Reynolds(d_tubo, vel, pho, vis):
        return (d_tubo * vel * pho) / vis

    @staticmethod
    def CD_max(lo, do):  # valid dor lo/do from 2 to 10, high Re numbers
        return 0.827 - 0.0085 * (lo / do)

    @staticmethod
    def CD_max_naka(lo, do):  # valid for lo/do from 1.5 to 17, high Re numbers
        return 0.868 - 0.0425 * ((lo / do) ** 0.5)

    @staticmethod
    def CD_Nakayama(lo, do, Re):  # Re 550 to 7000, lo/do 1.5 to 17
        termo1 = Re ** (5 / 6)
        termo2 = 17.11 * (lo / do) + 1.65 * (Re ** 0.8)
        return termo1 / termo2

    @staticmethod
    def CD_Asihmin(lo, do, Re):  # Re 100 to 150000, lo/do 2 to 5
        termo1 = lo / do
        termo2 = (58 * termo1) / Re
        termo3 = 1.23 + termo2
        return 1 / termo3

    def CD_Lichtarowicz(self, lo, do, Re):  # Re 10 to 20000, lo/do 2 to 10
        termo1 = (20 / Re) * (1 + 2.25 * (lo / do))
        termo2 = 1 / self.CD_max(lo, do)
        termo3 = termo1 + termo2
        return 1 / termo3

