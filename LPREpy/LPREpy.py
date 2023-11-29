from CEApy import CEA
import logging as log
import numpy as np
log.basicConfig(level=log.INFO)


class LPREpy:
    def __init__(self, name="LPREpy_Engine", dx=0.001, iterations=1000, error=10 ** (-5), output_geo_len_uni='mm'):
        log.info("\n\nLPRE - Analysis of : {}".format(name))
        log.info("Liquid Propellant Rocket Engine Python Library (LPREpy) - Design of LREs")
        log.info("LPREpy: checkout instructions at: https://github.com/juliomachad0/LPREpy")
        log.info("LPREpy: Input units always in SI")
        self.name = name

        # Variáveis de cálculo da geometria
        self.geometry_length_unit = output_geo_len_uni
        self.current_unit = None
        self.dx = dx  # passo geometrico
        self.error = error  # erro na construção do perfil
        self.ite = iterations  # iterações na construção do perfil

        # parametros
        self.F = None  # nominal thrust, N
        self.pc = None  # pressão da câmara, Pa
        self.aeat = None  # taxa de expansão do nozzle
        self.acat = None  # taxa de contração da câmara
        self.of = None  # razão de mistura
        self.pe = 0  # exit pressure - pe
        # condition of pe: default (0 - vacuum), user defined or calculated by CEA
        self.pe_condi = 'default'
        self.cf = None  # thrust coefficient
        self.efi_cf = 0.98  # eficiência do coeficiente de empuxo
        self.cf_esti = 0
        # chamber geometry parameters
        self.lc = 0  # chamber total length
        self.lcil = 0  # length of cylindrical part of the chamber
        self.lconv = 0  # length of the convergent part of the chamber, H + h
        self.H = 0  # length of the R2 arc projected in the symmetry axis
        self.h = 0  # length of the R1 arc projected in the symmetry axis
        self.y = 0  # height of the junction between R2 and R1 arc
        self.kappa = 1.5  # ratio of the R2 and R1 arc, kappa = R2/R1
        self.Lcarac = 0  # characteristic length
        # main diameters
        self.dc = 0  # chamber diameter
        self.dt = 0  # throat diameter
        self.de = 0  # nozzle exit diameter
        # flow rate
        self.M_dot_total = None
        self.M_dot_ox = None
        self.M_dot_fuel = None
        # nozzle geometry parameters
        self.Tn = 0.2  # Tn = Rn/Rt
        self.Rn = 0
        self.gam_ari_mean = 0
        self.gam_log_mean = 0
        self.gam_exit = 0
        self.gam_main = 0
        self.method_nozzle = 'MOC'
        self.moccond = 'aeat'
        self.gam_condi = 'log'
        self.res_aeat = 0

        # CEA
        self.CEA = CEA(name=self.name)
        #
        self.dataframecea = None
        #
        self.isp = None
        self.ivac = None
        self.gam = None
        self.p = None
        self.rho = None
        self.Mach_number = None
        self.son = None
        self.vel = None
        self.T = None

        # Inner Contour
        self.InnerContour = InnerProfile()
        self.axis_chamber, self.profile_chamber = None, None
        self.axis_nozzle, self.profile_nozzle = None, None
        self.axis_inner_contour, self.profile_inner_contour = None, None

    def __CSA(self):
        cf_exit = self.cf[2]
        self.cf_esti = cf_exit * self.efi_cf  # coeficiente de empuxo estimado
        if self.pe_condi == 'CEA':
            self.pe = self.p[2]
        elif self.pe_condi == 'user':
            pass
        else:
            self.pe_condi = 'default'
            self.pe = 0

        At = self.F / (self.pc * self.cf_esti - self.aeat * self.pe)  # metros, area da garganta
        Ae = self.aeat * At  # metros, área de expansão

        self.dt = 2 * np.sqrt(At / np.pi)  # metros, diametro da garganta
        self.dc = self.dt * np.sqrt(self.acat)  # metros, diametro da parte cilindrica
        self.de = 2 * np.sqrt(Ae / np.pi)  # metros, diametro da saida

    def __M_dot(self):
        self.M_dot_total = self.F/(self.isp[2]*9.80665)
        self.M_dot_ox = (self.of/(self.of+1))*self.M_dot_total
        self.M_dot_fuel = (1/(self.of+1))*self.M_dot_total

    def Initial_Parameters(self, F, pc, aeat, acat, of, efi_cf=0.98, pe: "exit pressure" = None, lc=0):
        self.F = F
        self.pc = pc
        self.aeat = aeat
        self.acat = acat
        self.of = of
        self.efi_cf = efi_cf
        self.lc = lc

        if pe is None:
            self.pe_condi = 'default'
            self.pe = 0
            log.info("LPREpy: value of pe (exit pressure) parameter: {:.4f} Pa".format(self.pe))
        elif type(pe) is str:
            if pe.lower() == 'cea':
                self.pe_condi = 'CEA'
            elif pe.lower() == 'default':
                self.pe_condi = 'default'
                self.pe = 0
            else:
                log.warning(
                    "LPREpy: In 'pe' parameter put 'CEA' or 'default (vacuum) 'a user defined value or let unfilled.")
                self.pe = 0
                log.info("LPREpy: value of pe (exit pressure) parameter: {:.4f} Pa".format(self.pe))
        elif (pe != 0) and (isinstance(pe, int) or isinstance(pe, float)):
            self.pe_condi = 'user'
            self.pe = pe
        else:
            log.warning("LPREpy: In 'pe' parameter put 'CEA' or a user defined value or let unfilled.")
            self.pe = 0
            log.info("LPREpy: value of pe (exit pressure) parameter: {:.4f} Pa".format(self.pe))

    def Input_Propellant_CEA_config(self, oxi, fuel, frozen='no', freezing_point='exit', equilibrium='yes',
                                    short='yes', include_acat='no', show_output_CEAfiles='no'):
        log.info("LPREpy: This method uses CEApy library, \nwhich for now is only available in windows systems :/")

        # using CEApy and configuring the analysis
        self.CEA.settings(frozen=frozen, freezing_point=freezing_point, equilibrium=equilibrium, short=short, )
        self.CEA.input_propellants(oxid=oxi, fuel=fuel)
        self.CEA.output_parameters(['aeat', 'cf', 'p', 'isp', 'rho', 'ivac', 'gam',
                                    'mach', 't', 'M', 'cp', 'cond', 'vis', 'pran', 'son'])
        pc = self.pc / 100000  # converting from Pascal to bar
        if include_acat == 'no':
            self.CEA.input_parameters(sup_aeat=[self.aeat], chamber_pressure=[pc], of_ratio=[self.of])
        elif include_acat == 'yes':
            self.CEA.input_parameters(acat=[self.acat], sup_aeat=[self.aeat], chamber_pressure=[pc],
                                      of_ratio=[self.of])
        else:
            log.warning("LPREpy: I. include_acat (include acat in CEA analysis or not) must be 'yes' or 'no'.")
            log.warning("LPREpy:II. acat (contraction ratio of chamber convergent section) parameter not included")
            self.CEA.input_parameters(sup_aeat=[self.aeat], chamber_pressure=[pc], of_ratio=[self.of])

        # running CEA and getting results
        self.CEA.run()
        if show_output_CEAfiles.lower() == 'yes':
            self.CEA.show_out_file()
        elif show_output_CEAfiles.lower() == 'no':
            pass
        else:
            log.warning("show_output_CEAfiles must be 'yes' or 'no'")
        # getting cea parameters
        if include_acat == 'no':
            outcea = self.CEA.get_results('all', 'all')
        elif include_acat == 'yes':
            outcea_acat = self.CEA.get_results('all', 'all')
            outcea = outcea_acat.drop(0)
            outcea = outcea.reset_index()
        else:
            log.warning("'include_acat' parameter must be 'yes' or 'no'. 'no' considered.")
            outcea = self.CEA.get_results('all', 'all')


        if outcea is None:
            log.error("LPREpy: something went wrong, CEA analysis has empty results, analysis failed.")
        else:
            outcea['isp'] = outcea['isp'] / 9.80665
            outcea['ivac'] = outcea['ivac'] / 9.80665
            self.CEA.remove_analysis_file(name=self.name)
            self.isp = [outcea['isp'][0], outcea['isp'][1], outcea['isp'][2]]
            self.ivac = [outcea['ivac'][0], outcea['ivac'][1], outcea['ivac'][2]]
            self.cf = [outcea['cf'][0], outcea['cf'][1], outcea['cf'][2]]  # cf no nozzle
            self.p = [outcea['p'][0], outcea['p'][1], outcea['p'][2]] * 100000  # Pressao na saída
            self.gam = [outcea['gam'][0], outcea['gam'][1], outcea['gam'][2]]  # gamma na camara
            self.rho = [outcea['rho'][0], outcea['rho'][1], outcea['rho'][2]]
            self.Mach_number = [outcea['mach'][0], outcea['mach'][1], outcea['mach'][2]]  # mach na saida
            self.son = [outcea['son'][0], outcea['son'][1], outcea['son'][2]]
            self.vel = [self.Mach_number[0] * self.son[0], self.Mach_number[1] * self.son[1], self.Mach_number[2] * self.son[2]]
            #outcea['vel'] = self.vel  # velocity = sonic_velocity * mach_number
            self.dataframecea = outcea
            # calculating dc, dt and de
            self.__CSA()
            self.__M_dot()
            # mean values of gammas and at exit section
            self.gam_exit = self.gam[2]
            self.gam_log_mean = np.log(self.p[1] / self.p[2]) / np.log(self.rho[1] / self.rho[2])  # gam log mean
            self.gam_ari_mean = (self.gam[0] + self.gam[1] + self.gam[2]) / 3  # gam ari mean

        # pass

    def Input_CEA_parameters(self, cea: list, format_data: "Pandas_df df or list" = "Pandas"):
        if format_data.lower() == 'pandas':
            cea['isp'] = cea['isp'] / 9.80665
            cea['ivac'] = cea['ivac'] / 9.80665
            self.isp = [cea['isp'][0], cea['isp'][1], cea['isp'][2]]
            self.ivac = [cea['ivac'][0], cea['ivac'][1], cea['ivac'][2]]
            self.cf = [cea['cf'][0], cea['cf'][1], cea['cf'][2]]  # cf no nozzle
            self.p = [cea['p'][0], cea['p'][1], cea['p'][2]] * 100000  # Pressao na saída
            self.gam = [cea['gam'][0], cea['gam'][1], cea['gam'][2]]  # gamma na camara
            self.rho = [cea['rho'][0], cea['rho'][1], cea['rho'][2]]
            self.Mach_number = [cea['mach'][0], cea['mach'][1], cea['mach'][2]]  # mach na saida
            self.son = [cea['son'][0], cea['son'][1], cea['son'][2]]
        elif format_data.lower() == 'list':
            # cea list: [isp, ivac,cf,p,gam,rho,M,son,vel]
            self.isp = cea[0]
            self.ivac = cea[1]
            self.cf = cea[2]
            self.p = cea[3] * 100000  # pressure
            self.gam = cea[4]  # gamma
            self.rho = cea[5]
            self.Mach_number = cea[6]  # mach number
            self.son = cea[7]
        else:
            log.error("LPREpy: 'format' parameter must be 'Pandas' (Pandas df result from CEA library)"
                      "\nor list in shape: [[a(chamber), b(throat), c(exit)],[d, e, f],[g, h, i]...]\n"
                      "sequence of list: [isp,isp_vac,cf,chamber_pressure,gamma,rho,Mach,son_vel,velocity]\n"
                      "isp=[isp chamber,isp throat,isp exit], gamma =[gamma chamber, gamma throat, gamma exit], ...")
            return 'error'
        # relative velocity of combustion product gases
        self.vel = [self.Mach_number[0] * self.son[0], self.Mach_number[1] * self.son[1], self.Mach_number[2] * self.son[2]]
        cea['vel'] = self.vel
        self.dataframecea = cea
        # calculating dc, dt and de
        self.__CSA()
        self.__M_dot()
        # mean values of gammas and at exit section
        self.gam_exit = self.gam[2]  # nozzle exit section
        self.gam_log_mean = np.log(self.p[1] / self.p[2]) / np.log(self.rho[1] / self.rho[2])  # gam log mean
        self.gam_ari_mean = (self.gam[0] + self.gam[1] + self.gam[2]) / 3  # gam ari mean

    def Chamber_Profile(self, kappa=None, Lcarac=None, dc=None, dt=None):
        if kappa is None:
            log.info("LPREpy: value of kappa parameter: {:.2f}".format(self.kappa))
        else:
            self.kappa = kappa

        if (Lcarac is None) and (self.Lcarac == 0):
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
            return
        elif isinstance(Lcarac, str):
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
            return
        elif (Lcarac is None) and (self.Lcarac != 0):
            log.info("LPREpy: value of characteristic length: {:.4f} m".format(self.Lcarac))
        elif (isinstance(Lcarac, float) or isinstance(Lcarac, int)) and Lcarac != 0:
            self.Lcarac = Lcarac
            log.info("LPREpy: new value of characteristic length: {:.4f} m".format(self.Lcarac))
        else:
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
            return

        if dc is None:
            log.info("LPREpy: Value of chamber diameter: {:.4f} MM".format(self.dc * 1000))
        else:
            self.dc = dc

        if dt is None:
            log.info("LPREpy: Value of throat diameter: {:.4f} MM".format(self.dt * 1000))
        else:
            self.dt = dt

        self.axis_chamber, self.profile_chamber = self.InnerContour.Chamber_Profile(dc=self.dc, dt=self.dt,
                                                                                    Lcarac=self.Lcarac,
                                                                                    kappa=self.kappa, dx=self.dx,
                                                                                    ite=self.ite, erro=self.error)
        return self.axis_chamber, self.profile_chamber

    def Nozzle_Profile(self, kappa=None, Lcarac=None, dc=None, dt=None, gam=None, Tn=None,
                       moccond='aeat', method='MOC', gam_cond='log'):
        if kappa is None:
            log.info("LPREpy: Kappa parameter in default value: {:.2f}".format(self.kappa))
        else:
            self.kappa = kappa

        if (Lcarac is None) and (self.Lcarac == 0):
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
        elif (Lcarac is None) and (self.Lcarac != 0):
            log.info("LPREpy: value of characteristic length: {:.4f} m".format(self.Lcarac))
        else:
            self.Lcarac = Lcarac
            log.info("LPREpy: value of characteristic length: {:.4f} m".format(self.Lcarac))

        if dc is None:
            log.info("LPREpy: Value of chamber diameter: {:.4f} MM".format(self.dc * 1000))
        else:
            self.dc = dc

        if dt is None:
            log.info("LPREpy: Value of throat diameter: {:.4f} MM".format(self.dt * 1000))
        else:
            self.dt = dt

        if gam is None:
            if gam_cond == 'log_mean':
                self.gam_main = self.gam_log_mean
                log.info("LPREpy: value of gamma parameter (log mean): {:.4f}".format(self.gam_main))
            elif gam_cond == 'ari_mean':
                self.gam_main = self.gam_ari_mean
                log.info("LPREpy: value of gamma parameter (ari. mean): {:.4f}".format(self.gam_main))
            elif gam_cond == 'gam_exit':
                self.gam_main = self.gam_exit
                log.info("LPREpy: value of gamma parameter (gam at exit)".format(self.gam_main))
            else:
                log.warning("LPREpy: value of gam_cond parameter must be None (default - log_mean), "
                            "'log_mean', 'ari_mean' or 'gam_exit'")
                self.gam_main = self.gam_log_mean
                log.info("LPREpy: value of gamma parameter (default)".format(self.gam_main))
        elif ((type(gam) is float) or (type(gam) is int)) and (gam != 0):
            self.gam_main = gam
        else:
            log.warning("LPREpy: type of gam parameter must be Float")
            log.warning("LPREpy: value of gam_cond parameter must be None (default - log_mean), "
                        "'log_mean', 'ari_mean' or 'gam_exit'")
            self.gam_condi = 'log_mean'
            self.gam_main = self.gam_log_mean
            log.info("LPREpy: value of gamma parameter (ari. mean)".format(self.gam_main))

        if Tn is None:
            log.info("LPREpy: Tn (Rn/Rt) parameter in default value: {:.2f}".format(self.Tn))
        elif (type(Tn) is float) or (type(Tn) is int):
            self.Tn = Tn
        else:
            log.warning("LPREpy: Tn (Rn/Rt) parameter must be None (default) or float")
            log.info("LPREpy: Tn (Rn/Rt) parameter in default value: {:.2f}".format(self.Tn))

        if moccond == 'aeat':
            self.moccond = 'aeat'
        elif moccond == 'min':
            self.moccond = 'min'
        else:
            log.warning("moc cond must be 'aeat' to achieve the provided expansion rate aeat or 'min' to build the \n"
                        "minimum moc profile without chocks inside the nozzle")
            self.moccond = 'aeat'
            log.info("LPREpy: condition of the MOC nozzle profile construction: {} ".format(self.moccond))

        if method == 'MOC':
            self.method_nozzle = 'MOC'
        else:
            self.method_nozzle = 'MOC'
            log.warning("Construction method of the nozzle profile: {}".format(self.method_nozzle))
        self.axis_nozzle, self.profile_nozzle = self.InnerContour.Nozzle_Profile(dc=self.dc,
                                                                                 dt=self.dt,
                                                                                 Lcarac=self.Lcarac,
                                                                                 kappa=self.kappa,
                                                                                 gam=self.gam_main,
                                                                                 Me=self.Mach_number[2],
                                                                                 aeat=self.aeat,
                                                                                 dx=self.dx,
                                                                                 ite=self.ite,
                                                                                 erro=self.error,
                                                                                 Tn=self.Tn,
                                                                                 moccond=self.moccond,
                                                                                 method=self.method_nozzle)
        return self.axis_nozzle, self.profile_nozzle

    def Inner_Profile(self, kappa=None, Lcarac=None, dc=None, dt=None, gam=None, Tn=None,
                      moccond='aeat', method='MOC', gam_cond='log_mean', uni='MM'):
        if kappa is None:
            log.info("LPREpy: Kappa parameter in default value: {:.2f}".format(self.kappa))
        else:
            self.kappa = kappa

        if (Lcarac is None) and (self.Lcarac == 0):
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
            return 'error'
        elif (isinstance(Lcarac, float) or isinstance(Lcarac, int)) and Lcarac != 0:
            self.Lcarac = Lcarac
            log.info("LPREpy: value of characteristic length: {:.4f} m".format(self.Lcarac))
        else:
            log.error("LPREpy: Lcarac need a value provided by the user or calculated by the injector functions")
            log.info("LPREpy: Actual value of characteristic length: {:.4f} m".format(self.Lcarac))
            return 'error'

        if dc is None:
            log.info("LPREpy: Value of chamber diameter: {:.4f} MM".format(self.dc * 1000))
        else:
            self.dc = dc

        if dt is None:
            log.info("LPREpy: Value of throat diameter: {:.4f} MM".format(self.dt * 1000))
        else:
            self.dt = dt

        if gam is None:
            if gam_cond == 'log_mean':
                self.gam_main = self.gam_log_mean
                log.info("LPREpy: value of gamma parameter (log mean): {:.4f}".format(self.gam_main))
            elif gam_cond == 'ari_mean':
                self.gam_main = self.gam_ari_mean
                log.info("LPREpy: value of gamma parameter (ari. mean): {:.4f}".format(self.gam_main))
            elif gam_cond == 'gam_exit':
                self.gam_main = self.gam_exit
                log.info("LPREpy: value of gamma parameter (gamma at exit): {:.4f}".format(self.gam_main))
            else:
                log.warning("LPREpy: value of gam_cond parameter must be None (default - log_mean), "
                            "'log_mean', 'ari_mean' or 'gam_exit'")
                self.gam_main = self.gam_log_mean
                log.info("LPREpy: value of gamma parameter (default): {:.4f}".format(self.gam_main))
        elif ((type(gam) is float) or (type(gam) is int)) and (gam != 0):
            self.gam_main = gam
            log.info("LPREpy: value of gamma parameter (user defined)".format(self.gam_main))
        else:
            log.warning("LPREpy: type of gam parameter must be Float")
            log.warning("LPREpy: value of gam_cond parameter must be None (default - log_mean), "
                        "'log_mean', 'ari_mean' or 'gam_exit'")
            self.gam_condi = 'log_mean'
            self.gam_main = self.gam_log_mean
            log.info("LPREpy: value of gamma parameter (ari. mean): {:.4f}".format(self.gam_main))

        if Tn is None:
            log.info("LPREpy: Tn (Rn/Rt) parameter in default value: {:.2f}".format(self.Tn))
        elif (type(Tn) is float) or (type(Tn) is int):
            self.Tn = Tn
        else:
            log.warning("LPREpy: Tn (Rn/Rt) parameter must be None (default) or float")
            log.info("LPREpy: Tn (Rn/Rt) parameter in default value: {:.2f}".format(self.Tn))

        if moccond == 'aeat':
            self.moccond = 'aeat'
        elif moccond == 'min':
            self.moccond = 'min'
        else:
            log.warning("moc cond must be 'aeat' to achieve the provided expansion rate aeat or 'min' to build the \n"
                        "minimum moc profile without chocks inside the nozzle")
            self.moccond = 'aeat'
            log.info("LPREpy: condition of the MOC nozzle profile construction: {} ".format(self.moccond))

        if method == 'MOC':
            self.method_nozzle = 'MOC'
        else:
            self.method_nozzle = 'MOC'
            log.warning("Construction method of the nozzle profile: {}".format(self.method_nozzle))

        self.geometry_length_unit = uni
        self.current_unit = uni

        self.axis_inner_contour, self.profile_inner_contour = self.InnerContour.Inner_Profile(
            dc=self.dc, dt=self.dt, Lcarac=self.Lcarac, kappa=self.kappa,
            gam=self.gam_main, Me=self.Mach_number[2], aeat=self.aeat, dx=self.dx,
            ite=self.ite, erro=self.error, Tn=self.Tn,
            method=self.method_nozzle, moccond=self.moccond,
            uni=self.geometry_length_unit
        )

        return self.axis_inner_contour, self.profile_inner_contour

    def Info_GeometricMassFlowParams(self):
        log.info("\n\nLPREpy: Geometric and Mass Flow Parameters")
        log.info("LPREpy: Engine: {}".format(self.name))
        log.info("LPREpy: Pc: {:.2f} bar, F: {:2f} N, o/f: {:2f}".format(self.pc/100000, self.F, self.of))
        log.info("LPREpy: aeat: {:.2f}, acat: {:.2f}, pe: {:.2f} bar".format(
            self.aeat, self.acat, self.pe/100000))
        log.info("LPREpy: dc: {:2f} mm, dt: {:2f} mm, de: {:.2f} mm".format(
            self.dc*1000, self.dt*1000, self.de*1000
        ))
        log.info("LPREpy: m_dot_total: {:.2f} g/s, m_dot_ox: {:.2f} g/s, m_dot_fuel: {:.2f} g/s".format(
            self.M_dot_total*1000, self.M_dot_ox*1000, self.M_dot_fuel*1000
        ))

    def Info_CEA_Results(self):
        log.info("\n\nLPREpy: CEA Results")
        log.info("index: 0-chamber, 1-throat, 2-exit section")
        log.info(self.dataframecea)

    def Get_results(self, diameters_Mflow: str='no'):
        if diameters_Mflow.lower() == 'yes':
            return [self.dc, self.dt, self.de, self.M_dot_total, self.M_dot_ox, self.M_dot_fuel]

    def Get_CEA_data(self, parameters: str = 'all', section: str = 'all'):
        all_cea_parameters = [
            'p', 't', 'rho', 'h', 'u', 'g', 's', 'm', 'mw', 'cp', 'gam', 'son',  # thermo prop
            'pipe', 'mach', 'aeat', 'cf', 'ivac', 'isp',  # rocket performance
            'vis', 'cond', 'condfz', 'pran', 'pranfz',  # transport properties
            '%f', 'o/f', 'phi,eq.ratio', 'r,eq.ratio']  # fuel-oxidant mixture parameters

        sections = ['chamber', 'throat', 'exit', 'all']
        if section not in sections:
            log.error("LPREpy: 'section' parameter must be 'chamber', 'throat' or 'exit'")
            return

        if parameters.lower() == 'all':
            if section.lower() == 'all':
                return self.dataframecea
            elif section.lower() == 'chamber':
                return self.dataframecea.loc[0]
            elif section.lower() == 'throat':
                return self.dataframecea[1]
            elif section.lower() == 'exit':
                return self.dataframecea[2]

        elif parameters.lower() in all_cea_parameters:
            if section.lower() == 'all':
                return self.dataframecea[parameters.lower()]
            elif section.lower() == 'chamber':
                return self.dataframecea[parameters.lower()][0]
            elif section.lower() == 'throat':
                return self.dataframecea[parameters.lower()][1]
            else:
                return self.dataframecea[parameters.lower()][2] #  exit

class InnerProfile:
    def __init__(self):
        self.chamber_axis = None
        self.chamber_profile = None
        self.rn_axis = None
        self.rn_profile = None
        self.div_axis = None
        self.div_profile = None
        self.nozzle_axis = None
        self.nozzle_profile = None
        self.inner_contour_axis = None
        self.inner_contour_profile = None

    # ****************************************** ARCs functions ******************************************
    def Calc_Arcos(self, r, yo, yf, H, xo, dx, quadrante, iteracoes, erro):
        b = yo - r
        xo = xo + dx
        xf = xo + H
        eixo = np.arange(xo, xf, dx)
        eixo = eixo.tolist()
        a = self.aprox_inicial(xo, r, b, yf, eixo, 1)
        a1 = self.newtonraphson(a, yo, yf, r, xo, xf, 'fa1', iteracoes, erro)
        b1 = self.funcaoB(yf, r, xf, a1, 'fb1')
        b2 = self.funcaoB(yf, r, xf, a1, 'fb2')
        a2 = self.newtonraphson(a, yo, yf, r, xo, xf, 'fa2', iteracoes, erro)
        b3 = self.funcaoB(yf, r, xf, a2, 'fb1')
        b4 = self.funcaoB(yf, r, xf, a2, 'fb2')
        E1 = self.Calc_Erro(r, a1, b1, xo, xf, yo, yf, 1)
        E2 = self.Calc_Erro(r, a1, b2, xo, xf, yo, yf, 1)
        E3 = self.Calc_Erro(r, a2, b3, xo, xf, yo, yf, 1)
        E4 = self.Calc_Erro(r, a2, b4, xo, xf, yo, yf, 1)
        menorerro = min([E1, E2, E3, E4])
        if menorerro == E1:
            a = a1
            b = b1
        elif menorerro == E2:
            a = a1
            b = b2
        elif menorerro == E3:
            a = a2
            b = b3
        elif menorerro == E4:
            a = a2
            b = b4
        eixo, perfil = self.translacao_quadrante(r, a, b, xo, xf, yo, yf, dx, eixo, quadrante)
        return eixo, perfil

    # funcao de translacao dos quadrantes
    def translacao_quadrante(self, r, a, b, xo, xf, yo, yf, dx, eixo, quadrante):
        mod1 = abs(xf - a)
        mod2 = abs(xo - a)
        if yo > yf:
            if quadrante == 1:
                eixo, perfil = self.circulo(r, a, b, eixo, quadrante)
                return eixo, perfil
            if quadrante == 2:
                # reflexão do eixo
                eixo1 = np.arange(a - mod1, a - mod2, dx)
                eixo1, perfil = self.circulo(r, a, b, eixo1, 1)
                return eixo, perfil
            if quadrante == 3:
                # calculo do perfil 1
                eixo, perfil1 = self.circulo(r, a, b, eixo, 1)
                # reflexao do eixo
                eixo1 = np.arange(a - mod1, a - mod2, dx)
                # calculo da nova curva
                eixo1, perfil2 = self.circulo(r, a, b, eixo1, 3)
                # translação da curva
                omega = abs(perfil1[0] - perfil2[0])
                perfil3 = list(map(lambda x: x + omega, perfil2))
                return eixo, perfil3
            if quadrante == 4:
                # calculo do perfil 1
                eixo, perfil1 = self.circulo(r, a, b, eixo, 1)
                # calculo do perfil 2
                eixo, perfil2 = self.circulo(r, a, b, eixo, 4)
                # translação da curva
                omega = abs(perfil1[-1] - perfil2[0])
                perfil3 = list(map(lambda x: x + omega, perfil2))
                return eixo, perfil3
        if yo < yf:
            if quadrante == 4:
                # trecho de codigo que precisa de checagem - inicio
                gama = a + mod1
                lbda = a + mod2
                if gama < lbda:
                    eixo1 = np.arange(gama, lbda, dx)
                else:
                    eixo1 = np.arange(lbda, gama, dx)
                # trecho de codigo que precisa de checagem - fim

                # calculo do perfil 1
                eixo1, perfil1 = self.circulo(r, a, b, eixo1, 1)
                eixo1 = eixo1.tolist()
                # calculo do perfil 2
                eixo2, perfil2 = self.circulo(r, a, b, eixo1, 4)
                # translação da curva
                omega = abs(perfil1[-1] - perfil2[0])
                perfil3 = list(map(lambda x: x + omega, perfil2))
                return eixo1, perfil3

    # funcao de newton-raphson
    @staticmethod
    def funcaoA(a, Rc, y, r, xo, xf, condicao):
        # pp = np.sqrt((r**2)+((xf-a)**2))

        pp = (xf - a) ** 2

        raiz = (r ** 2) - ((xo - a) ** 2)

        if raiz <= 0:
            if condicao == 'fa1':
                tp = (y - Rc) ** 2
            elif condicao == 'fa2':
                tp = (y - Rc) ** 2
            funcao = pp + tp - (r ** 2)
            return funcao
        else:
            sp = np.sqrt(raiz)
            if condicao == 'fa1':
                tp = (y - Rc + sp) ** 2
            elif condicao == 'fa2':
                tp = (y - Rc - sp) ** 2
            funcao = pp + tp - (r ** 2)
            return funcao

    # diferencial da funcao de newton-raphson
    @staticmethod
    def dif_funcaoA(a, Rc, y, r, xo, xf, condicao):

        pp = 2 * (a - xf)
        raiz = (r ** 2) - ((xo - a) ** 2)

        if raiz <= 0:
            return pp  # pp # derivada = pp
        else:
            sp = np.sqrt(raiz)
            tp = xo - a
            if condicao == 'fa1':
                qp = 2 * (y - Rc + sp) / sp
                derivada = pp + qp * tp
                return derivada
            elif condicao == 'fa2':
                qp = -2 * (y - Rc - sp) / sp
                derivada = pp + qp * tp
                return derivada

    def newtonraphson(self, a, Rc, y, r, xo, xf, condicao, iteracoes, erro):
        i = 0
        while i <= iteracoes:
            xnext = a - self.funcaoA(a, Rc, y, r, xo, xf, condicao) / self.dif_funcaoA(a, Rc, y, r, xo, xf, condicao)
            a = xnext
            raiz = abs(self.funcaoA(a, Rc, y, r, xo, xf, condicao))
            if raiz < erro:
                return a
            i += 1
        return a

    # função de calculo de b
    @staticmethod
    def funcaoB(y, r, xf, a, condicao):
        sp = np.sqrt((r ** 2) - ((xf - a) ** 2))
        if condicao == 'fb1':
            b = y + sp
            return b
        elif condicao == 'fb2':
            b = y - sp
            return b
        else:
            # print("Informe 'fb1' ou 'fb2'")
            return

    # funcao de aproximação inicial
    def aprox_inicial(self, a, r, b, y, eixo, quadrante):
        daxo = 0.5
        eixo, perfil = self.circulo(r, a, b, eixo, quadrante)  # calculando perfil
        taxa = y / perfil[-1]  # calculando taxa inicial
        while taxa < 1:
            a = a - daxo  # variando a de xo até af
            eixo, perfil = self.circulo(r, a, b, eixo, quadrante)  # calculando perfil
            taxa = y / perfil[-1]  # calculando taxa
            # variando passo de aproximação
            if taxa > 0.5:
                daxo = 0.1
            if taxa > 0.95:
                daxo = 0.01
            if taxa > 0.98:
                daxo = 0.001
            if taxa > 0.999:
                daxo = 0.0001
        return a

    # calculo do erro
    def Calc_Erro(self, r, a, b, xo, xf, Rc, y, quadrante):
        y1 = self.pontoCirculo(r, a, b, xo, quadrante)
        y2 = self.pontoCirculo(r, a, b, xf, quadrante)
        e1 = abs(((y1 - Rc) / Rc) * 100)
        e2 = abs(((y2 - y) / y) * 100)
        return e1 + e2

    # calculo de ponto do circulo
    @staticmethod
    def pontoCirculo(r, a, b, x, quadrante):
        if (quadrante == 1) or (quadrante == 2):
            return np.sqrt(abs((r ** 2) - ((x - a) ** 2))) + b
        elif (quadrante == 3) or (quadrante == 4):
            return -np.sqrt(abs((r ** 2) - ((x - a) ** 2))) + b
        else:
            # print('Informe 1, 2, 3 ou  na opção quadrante. ')
            return

    # funcao de calculo do circulo
    @staticmethod
    def circulo(r, a, b, eixo, quadrante):
        if (quadrante == 1) or (quadrante == 2):
            perfil = list(map(lambda x: np.sqrt((r ** 2) - ((x - a) ** 2)) + b, eixo))
            return eixo, perfil
        elif (quadrante == 3) or (quadrante == 4):
            perfil = list(map(lambda x: -np.sqrt((r ** 2) - ((x - a) ** 2)) + b, eixo))
            return eixo, perfil
        else:
            # print('Informe 1, 2, 3 ou  na opção quadrante. ')
            return

    # ****************************************** CHAMBER PROFILE FUNCTIONS ******************************************

    @staticmethod
    def Lc(dc, dt, Lcarac, kappa):
        Ac = (np.pi / 4) * np.power(dc, 2)  # metros, cross-sectional area of the cylindrical section
        At = (np.pi / 4) * np.power(dt, 2)  # metros, cross-sectional area of the throat section
        acat = Ac / At  # taxa de contração da câmara
        racat = np.sqrt(acat)  # raiz de acat
        # L convergente
        ldelta1 = np.power(2 + kappa * racat, 2)
        ldelta2 = np.power((kappa - 1) * racat + 3, 2)
        lconv = 0.5 * dt * np.sqrt(ldelta1 - ldelta2)  # metros
        # seções de suporte
        h = lconv * (2 / (2 + kappa * racat))
        H = lconv - h
        y = (dt / 2) * ((h * racat + H) / lconv)
        yast = (2 * y) / dt
        # L cilíndrico
        Dvc1 = ((2 * acat + np.power(yast, 2)) * H) / (3 * lconv)
        Dvc2 = ((np.power(yast, 2) + yast + 4) * h) / (6 * lconv)
        Dvc = At * lconv * (Dvc1 + Dvc2)
        Vc = Lcarac * At
        lcil = (Vc - Dvc) / Ac
        lc = lconv + lcil
        return h, H, y, lc, lconv, lcil

    @staticmethod
    def cilinder(Rc, lcil, dx):
        eixo_lcil = np.arange(0, lcil, dx)
        eixo_lcil = eixo_lcil.tolist()
        perfil_lcil = []
        for i in range(len(eixo_lcil)):
            perfil_lcil.append(Rc)
        return eixo_lcil, perfil_lcil

    def R1(self, R1, lcil, H, h, y, dx, ite, erro):
        # R1 = dt, xo_r1 = l_cil + H, f_xo_r1 = y, f_xf_r1 = Rt, length_x = h
        eixo, perfil = self.Calc_Arcos(R1, y, R1 / 2, h, lcil + H, dx, 3, ite, erro)
        return eixo, perfil

    def R2(self, kappa, dc, lcil, H, y, dx, ite, erro):
        # R2 = kappa*dc=kappa*2*Rc, xo_r2 = lcil, f_xo_r2 = Rc = dc/2, f_xf_r2 = y, length = H
        eixo, perfil = self.Calc_Arcos(kappa * dc, dc / 2, y, H, lcil, dx, 1, ite, erro)
        return eixo, perfil

    def Chamber_Profile(self, dc, dt, Lcarac, kappa, dx, ite, erro):
        h, H, y, lc, lconv, lcil = self.Lc(dc, dt, Lcarac, kappa)  # always on meters

        uni_c = 1000  # calculating in MM first
        # cylindrical profile
        eixo_lcil, perfil_lcil = self.cilinder((dc / 2) * uni_c, lcil * uni_c, dx)
        # R1 arc profile
        eixo_r1, perfil_r1 = self.R1(dt * uni_c, lcil * uni_c, H * uni_c, h * uni_c, y * uni_c, dx, ite, erro)
        # R2 arc profile
        eixo_r2, perfil_r2 = self.R2(kappa=kappa, dc=dc * uni_c, lcil=lcil * uni_c,
                                     H=H * uni_c, y=y * uni_c, dx=dx, ite=ite, erro=erro)
        # convergent section vector
        eixo_camara = eixo_lcil + eixo_r2 + eixo_r1
        perfil_camara = perfil_lcil + perfil_r2 + perfil_r1
        self.chamber_axis, self.chamber_profile = eixo_camara, perfil_camara
        return eixo_camara, perfil_camara

    #  ****************************************** NOZZLE PROFILE ******************************************

    # FUNÇÃO DE PRANDTL MEYER
    @staticmethod
    def PrtM(gam, mach):
        A = np.sqrt((gam + 1) / (gam - 1))
        B = np.sqrt(((gam - 1) / (gam + 1)) * ((mach ** 2) - 1))
        PrdtlMeyer = A * np.arctan(B) - np.arctan(np.sqrt((mach ** 2) - 1))
        return PrdtlMeyer

    # Rn Arc
    def Funcao_Rn(self, gam, Me, dt, Tn):
        Rn = Tn * dt  # Rn, baseado em kessaev, Rn = taxa * Dt
        Teta_max = 0.5 * self.PrtM(gam, Me)  # Radianos, Teta máximo, gama na saida, mach na saida
        Dx = Rn * np.sin(Teta_max)  # distancia horizontal medida a partir do inicio do arco
        Dy = Rn * (1 - np.cos(Teta_max))  # distancia vertical medida a partir do inicio do arco
        return Rn, Dx, Dy

    def Arco_Rn(self, xo, dx, gam, Me, dt, Tn, iteracoes, erro):
        Rt = dt / 2
        Rn, Dx, Dy = self.Funcao_Rn(gam, Me, dt, Tn)
        xo = xo + dx
        yo = Rt
        yf = Rt + Dy
        quadrante = 4
        eixo, perfil = self.Calc_Arcos(Rn, yo, yf, Dx, xo, dx, quadrante, iteracoes, erro)
        return eixo, perfil

    # FUNCAO DE METHOD OF CHARACTERISTICS WITH DELTA ANGLE
    def MOC_DeltaAngle(self, gam, Rt, yo, Me, DeltaAngle):
        # CONVERSAO ANGULOS
        RadToDeg = 180 / np.pi
        DegToRad = np.pi / 180
        # CALCULATE T_MAX
        Teta_max = 0.5 * self.PrtM(gam, Me) * RadToDeg

        # CALCULATING TETA, P AND SL
        Teta = []
        P = [0]
        SL = [0]
        for m in range(1, int(Teta_max) + 2):
            Teta.append((DeltaAngle + (m - 1)) * DegToRad)  # Teta_i
        for m in range(1, int(Teta_max) + 1):
            P.append(yo * np.tan(Teta[m]))  # X-AXIS POINTS
            SL.append(yo / P[m])
        SL.pop(0)

        # DIVIDE (delta slope)
        teta_max = Teta_max * DegToRad
        DTW = np.tan(teta_max) / (len(P) - 1)

        # FIRST WALL SECTION
        x = [(yo + SL[0] * P[0]) / (SL[0] - np.tan(teta_max))]
        y = [np.tan(teta_max) * x[0] + yo]

        # WALLS
        s = [np.tan(teta_max)]
        b = [yo]
        for k in range(1, len(P) - 1):
            s.append(np.tan(teta_max) - k * DTW)  # slope
            b.append(y[k - 1] - s[k] * x[k - 1])  # y-int
            x.append((b[k] + SL[k] * P[k]) / (SL[k] - s[k]))  # pontos x
            y.append(s[k] * x[k] + b[k])  # pontos y

        # ADDING INITIAL POINTS, 0 in x e Rt in y
        x.insert(0, 0)
        y.insert(0, yo)

        # EXPANSION RATIO
        Re = y[-1]
        Aaeat = (Re / Rt) ** 2

        return x, y, Aaeat

    # MAJOR METHOD OF CHARACTERISTICS FUNCTION
    def MOC(self, gam, Me, dt, Rn_rto, eratio_ref, iteracoes, moccond):
        # CONVERSAO ANGULOS
        RadToDeg = 180 / np.pi

        Teta_max = 0.5 * self.PrtM(gam, Me) * RadToDeg
        DeltaAngle = (90 - Teta_max) - np.floor(90 - Teta_max)
        dx = 0.1

        Rn, Dx, Dy = self.Funcao_Rn(gam, Me, dt, Rn_rto)
        yo = dt / 2 + Dy
        if moccond == 'min':
            x, y, eratio = self.MOC_DeltaAngle(gam, (dt / 2), yo, Me, DeltaAngle)
            return x, y, eratio
        elif moccond == 'aeat':
            x, y, eratio = self.MOC_DeltaAngle(gam, (dt / 2), yo, Me, DeltaAngle)
            c = 1
            taxa = eratio / eratio_ref
            while taxa < 1:
                if taxa > 0.8:
                    dx = 0.01
                if taxa > 0.95:
                    dx = 0.001
                if taxa > 0.999:
                    dx = 0.0001
                DeltaAngle = DeltaAngle + dx
                x, y, eratio = self.MOC_DeltaAngle(gam, (dt / 2), yo, Me, DeltaAngle)
                taxa = eratio / eratio_ref
                c += 1
                if c > iteracoes:
                    break
            return x, y, eratio
        else:
            print(
                "moccond (MOC profile creation condition) must be 'aeat' to achieve \n"
                "the desired/informed nozzle expansion rate"
                "\nor 'min' to calculate the minimum MOC without internal shocks")
            return False

    # NOZZLE PROFILE
    def Nozzle_Profile(self, dc, dt, Lcarac, kappa, gam, Me, aeat, dx, ite, erro, Tn, moccond, method='MOC'):
        h, H, y, lc, lconv, lcil = self.Lc(dc, dt, Lcarac, kappa)

        eixo_rn, perfil_rn = self.Arco_Rn(lc * 1000, dx, gam, Me, dt * 1000, Tn, ite, erro)

        if method == 'MOC':
            xmoc, ymoc, Aaeat_moc = self.MOC(gam, Me, dt * 1000, Tn, aeat, ite, moccond)

            ultimorn = eixo_rn[-1] + dx
            xmoc = list(map(lambda x: x + ultimorn, xmoc))

            # correcting possible rn wrong height
            rn_chamber_dif = abs(perfil_rn[0] - self.chamber_profile[-1])
            # adjusting initial height of rn to height of rt, and final height of rn to initial height of the divergent
            perfil_rn = list(map(lambda yi: yi - abs(rn_chamber_dif), perfil_rn))

            # adjusting beginning of the moc curve with the final of rn curve
            # inclination of rn curve
            tan_teta = (perfil_rn[-1] - perfil_rn[-2]) / (eixo_rn[-1] - eixo_rn[-2])
            # distance betwenn last y rn dot and first y moc dot
            h = ymoc[0] - perfil_rn[-1]
            # distance betwenn last x rn dot and first x moc dot
            total_d = xmoc[0] - eixo_rn[-1]
            # projection at x-axis of rn curve
            d_proj_xrn = h / tan_teta
            # distance to remove in xmoc
            x_sub_moc = total_d - d_proj_xrn
            # adjusting xmoc again
            xmoc = [i - x_sub_moc for i in xmoc]

            self.rn_axis, self.rn_profile = eixo_rn, perfil_rn
            self.div_axis, self.div_profile = xmoc, ymoc
            axis = eixo_rn + xmoc
            profile = perfil_rn + ymoc
            return axis, profile

    # ****************************************** INNER PROFILE ******************************************
    def Inner_Profile(self, dc, dt, Lcarac, kappa, gam, Me, aeat, dx, ite, erro, Tn, method, moccond, uni):

        axis_chamber, profile_chamber = self.Chamber_Profile(dc, dt, Lcarac, kappa, dx, ite, erro)

        axis_nozzle, profile_nozzle = self.Nozzle_Profile(dc, dt, Lcarac, kappa, gam, Me, aeat,
                                                          dx, ite, erro, Tn, moccond, method)

        axis_contour = axis_chamber + axis_nozzle
        profile_contour = profile_chamber + profile_nozzle

        if axis_contour[0] < 0:
            diff = abs(axis_contour[0])
            axis_contour = [i+diff for i in axis_contour]

        # units, choosing profile units, first calculations always in 'MM'
        uni_c = 1
        if uni == "MM":
            uni_c = 1
        if uni == "m":
            uni_c = 0.001
        if uni == "inch":
            uni_c = 0.0393701
        if uni == "ft":
            uni_c = 0.00328084

        axis_contour = [i * uni_c for i in axis_contour]
        profile_contour = [i * uni_c for i in profile_contour]

        return axis_contour, profile_contour
