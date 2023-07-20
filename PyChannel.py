import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import differential_evolution


''' START EXPERIMENTAL DATA LOAD '''

folder = "New-Datasets/"

# Activation dataset

dataset_act = pd.read_csv(folder+'Activation_M373I.csv', sep=',', skiprows=1)

act_data = dataset_act.to_numpy()

x_act_WT = act_data[:, 0]
y_act_WT = act_data[:, 1]
yerr_act_WT = act_data[:, 2]

# Inactivation dataset

dataset_inact = pd.read_csv(folder+"Inactivation_M373I.csv", sep=',', skiprows=1)

inact_data = dataset_inact.to_numpy()

x_inact_WT = inact_data[:, 0]
y_inact_WT = inact_data[:, 1]
yerr_inact_WT = inact_data[:, 2]

# CS recovery dataser

dataset_cs = pd.read_csv(folder+"CS-Inactivation_M373I.csv", sep=',', skiprows=1)

cs_data = dataset_cs.to_numpy()

x_cs_WT = cs_data[:, 0]
y_cs_WT = cs_data[:, 1]
yerr_cs_WT = cs_data[:, 2]

# Recovery dataset

dataset_rec = pd.read_csv(folder+"Recovery_M373I.csv", sep=',', skiprows=1)

rec_data = dataset_rec.to_numpy()

x_rec_WT = rec_data[:, 0]
y_rec_WT = rec_data[:, 1]
yerr_rec_WT = rec_data[:, 2]

''' END EXPERIMENTAL DATA LOAD '''

# Guess of initial conditions for the states
C0_0 = 0.4390
C1_0 = 0.2588
C2_0 = 0.0572
C3_0 = 0.0056
C4_0 = 0.0002
I0_0 = 0.0128
I1_0 = 0.0553
I2_0 = 0.0894
I3_0 = 0.0642
I4_0 = 0.0172
O_0 = 0.0001

# Pack up the initial conditions:

S0 = [C0_0, C1_0, C2_0, C3_0, C4_0, I0_0, I1_0, I2_0, I3_0, I4_0, O_0]

''' START PROTOCOLS '''


def SimAct(S, t, p):

    C0 = S[0]
    C1 = S[1]
    C2 = S[2]
    C3 = S[3]
    C4 = S[4]
    I0 = S[5]
    I1 = S[6]
    I2 = S[7]
    I3 = S[8]
    I4 = S[9]
    O = S[10]

    # constants

    T = 291.0  # K or 18 degree celsius
    e = 1.602176634 * (10**-19.0)  # C
    K_B = 1.380649 * (10**-23.0)  # J*K^-1

    exp_factor = (e/(K_B * T)) * (10**-3)

    # Voltage sequences

    V = 0.0  # mV

    if 0 <= t < 0.5:
        V = -90.0  # mV

    if 0.5 <= t < 0.55:
        V = p[11]  # Vtest

    if t >= 0.55:
        V = -50.0  # mV

    k_CI = p[8]  # 1.301e+02 #s^-1
    k_IC = p[9]  # 3.900e-01   #s^-1 0.20

    f = p[10]  # 4.315e-01 # 0.31

    # voltage dependent rate constants

    alpha = p[0] * np.exp(p[1] * (V * exp_factor))
    beta = p[2] * np.exp(-1.0 * p[3] * (V * exp_factor))
    k_CO = p[4] * np.exp(p[5] * (V * exp_factor))
    k_OC = p[6] * np.exp(-1.0 * p[7] * (V * exp_factor))

    # ODEs

    dC0dt = beta * C1 + (k_IC/(f**4.0)) * I0 - (k_CI*(f**4.0) + 4.0 * alpha) * C0
    dC1dt = 4.0 * alpha * C0 + 2.0 * beta * C2 + (k_IC/(f**3.0)) * I1 - (k_CI*(f**3.0) + beta + 3.0 * alpha) * C1
    dC2dt = 3.0 * alpha * C1 + 3.0 * beta * C3 + (k_IC/(f**2.0)) * I2 - (k_CI*(f**2.0) + 2.0 * beta + 2.0 * alpha) * C2
    dC3dt = 2.0 * alpha * C2 + 4.0 * beta * C4 + (k_IC/f) * I3 - (k_CI*f + 3.0 * beta + 1.0 * alpha) * C3
    dC4dt = 1.0 * alpha * C3 + k_OC * O + k_IC * I4 - (k_CI + k_CO + 4.0 * beta) * C4

    dI0dt = beta * f * I1 + (k_CI*(f**4.0)) * C0 - (k_IC/(f**4.0) + 4.0 * (alpha/f)) * I0
    dI1dt = 4.0 * (alpha/f) * I0 + 2.0 * beta * f * I2 + (k_CI*(f**3.0)) * C1 - (k_IC/(f**3.0) + beta * f + 3.0 * (alpha/f)) * I1
    dI2dt = 3.0 * (alpha/f) * I1 + 3.0 * beta * f * I3 + (k_CI*(f**2.0)) * C2 - (k_IC/(f**2.0) + 2.0 * beta * f + 2.0 * (alpha/f)) * I2
    dI3dt = 2.0 * (alpha/f) * I2 + 4.0 * beta * f * I4 + (k_CI*f) * C3 - (k_IC/f + 3.0 * beta * f + 1.0 * (alpha/f)) * I3
    dI4dt = 1.0 * (alpha/f) * I3 + k_CI * C4 - (k_IC + 4.0 * beta * f) * I4

    dOdt = k_CO * C4 - k_OC * O

    return (dC0dt, dC1dt, dC2dt, dC3dt, dC4dt, dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dOdt)


def Act_Protocol(max_V, DeltaV):
    Vhold = -90.0 #mV

    Vtest = np.linspace(Vhold,max_V,np.abs(int((max_V-Vhold)/DeltaV))+1)

    return Vtest

def Act_LS_func(x, teta):

    # conductance

    gK_max  = 33.2     # nS

    # capacitance
    Cm      = 1.0    # microF cm^-2

    #define activation sequence protocol

    Vmax = 60.0 #mV

    increment = 10.0 #mV

    Vtest = Act_Protocol(Vmax, increment)

    # Time discretiztion
    tini = 0.00

    tend = 3.00

    ttest_i = 0.50 #time at which you start record the current

    ttest_f = 0.55 #time at which you end record the current

    dt = 1e-5

    # time array
    t = np.arange(tini,tend+dt,dt)
    Npoints = len(t)

    # time array
    #t = np.linspace(tini,tend,Npoints)

    # prepare empty arrays
    Open_states = np.zeros((Npoints,len(Vtest)))

    max_conductance = np.zeros(len(Vtest))


    for i in range (0,len(Vtest)):

        gamma = np.append(teta, Vtest[i])

        f = lambda S,t: SimAct(S, t, gamma)

        r = odeint(f, S0, x)

        Open_states[:,i] = r[:,10]

        max_conductance[i] = gK_max * np.amax(r[int(ttest_i/dt):int(ttest_f/dt),10])


    g_gmax = (max_conductance)/(np.amax(max_conductance))

    return g_gmax

def SimInact(C, t, p):
    C0=C[0]
    C1=C[1]
    C2=C[2]
    C3=C[3]
    C4=C[4]
    I0=C[5]
    I1=C[6]
    I2=C[7]
    I3=C[8]
    I4=C[9]
    O=C[10]


    #constants

    T = 291.0 #K or 18 degree celsius
    e =  1.602176634 * (10**-19.0) # C
    K_B = 1.380649 * (10**-23.0) # J*K^-1

    exp_factor = (e/(K_B * T)) * (10**-3)

    #Voltage sequences

    V = 0.0 #mV

    if 0 <= t < 0.50:
        V=-90.0

    if 0.50 <= t < 1.50:
        V=p[11]

    if 1.50 <= t <= 2.50:
        V=60.0

    if t > 2.50 :
        V=60.0



    # wild type parameters
#    alpha_0 = 450.0 #s^-1
#    alpha_1 = 0.23 #s^-1

#    beta_0 = 2.0 #s^-1
#    beta_1 = 2.2 #s^-1

#    k_CO_0 = 160.0 #s^-1
#    k_CO_1 = 0.27 #s^-1

#    k_OC_0 = 245.0 #s^-1
#    k_OC_1 = 0.33 #s^-1

#    k_CI = 25.0 #s^-1
#    k_IC = 0.3 #s^-1

#    f = 0.37

    k_CI = p[8]

    k_IC = p[9]

    f = p[10]

    #voltage dependent rate constants

    alpha = p[0] * np.exp(p[1] * (V * exp_factor))
    beta = p[2] * np.exp(-1.0 * p[3] * (V * exp_factor))
    k_CO = p[4] * np.exp(p[5] * (V * exp_factor))
    k_OC = p[6] * np.exp(-1.0 * p[7] * (V * exp_factor))

    # ODEs

    dC0dt = beta * C1 + (k_IC/(f**4.0)) * I0 - (k_CI*(f**4.0) + 4.0 * alpha) * C0
    dC1dt = 4.0 * alpha * C0 + 2.0 * beta * C2 + (k_IC/(f**3.0)) * I1 - (k_CI*(f**3.0) + beta + 3.0 * alpha) * C1
    dC2dt = 3.0 * alpha * C1 + 3.0 * beta * C3 + (k_IC/(f**2.0)) * I2 - (k_CI*(f**2.0) + 2.0 * beta + 2.0 * alpha) * C2
    dC3dt = 2.0 * alpha * C2 + 4.0 * beta * C4 + (k_IC/f) * I3 - (k_CI*f + 3.0 * beta + 1.0 * alpha) * C3
    dC4dt = 1.0 * alpha * C3 + k_OC * O + k_IC * I4 - (k_CI + k_CO + 4.0 * beta) * C4

    dI0dt = beta * f * I1 + (k_CI*(f**4.0)) * C0 - (k_IC/(f**4.0) + 4.0 * (alpha/f)) * I0
    dI1dt = 4.0 * (alpha/f) * I0 + 2.0 * beta * f * I2 + (k_CI*(f**3.0)) * C1 - (k_IC/(f**3.0) + beta * f + 3.0 * (alpha/f)) * I1
    dI2dt = 3.0 * (alpha/f) * I1 + 3.0 * beta * f * I3 + (k_CI*(f**2.0)) * C2 - (k_IC/(f**2.0) + 2.0 * beta * f + 2.0 * (alpha/f)) * I2
    dI3dt = 2.0 * (alpha/f) * I2 + 4.0 * beta * f * I4 + (k_CI*f) * C3 - (k_IC/f + 3.0 * beta * f + 1.0 * (alpha/f)) * I3
    dI4dt = 1.0 * (alpha/f) * I3 + k_CI * C4 - (k_IC + 4.0 * beta * f) * I4

    dOdt = k_CO * C4 - k_OC * O

    return (dC0dt, dC1dt, dC2dt, dC3dt, dC4dt, dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dOdt)

def Inact_Protocol(max_V, DeltaV):
    Vhold = -90.0 #mV

    Vtest = np.linspace(Vhold,max_V,np.abs(int((max_V-Vhold)/DeltaV))+1)

    return Vtest #array of testing voltages

def Inact_LS_func(x, teta):

    # conductance parameters

    #EK      = 0.0    # mV
    gK_max  = 33.2     # nS

    # Assuming no leaking
    #EL      = 0.0    # mV
    #gL_max  = 0.0    # mS

    # Membrane capacitance
    Cm      = 1.0    # microF cm^-2


    Vdepo =  60.0 #mV

    Vhold = -90.0 #mV


    #define activation sequence protocol

    Vmax = 60.0 #mV

    increment = 10.0 #mV

    Vtest = Inact_Protocol(Vmax, increment)

    # Time discretiztion

    tini = 0.0

    tend = 3.0

    ttest = 1.5

    dt = 1e-5

    # time array
    t = np.arange(tini,tend+dt,dt)
    Npoints = len(t)

    # prepare empty arrays
    Open_states = np.zeros((Npoints,len(Vtest)))

    max_conductance = np.zeros(len(Vtest))

    max_currents = np.zeros(len(Vtest))

    for i in range (0,len(Vtest)):
        gamma = np.append(teta, Vtest[i])

        f = lambda S,t: SimInact(S, t, gamma)

        r = odeint(f, S0, x)

        Open_states[:,i] = r[:,10]

        max_conductance[i] = gK_max * np.amax(r[int(ttest/dt):,10]-r[int(ttest/dt)-1,10])

        max_currents[i] = max_conductance[i] * (Vdepo - Vhold)

    I_Imax = (max_currents)/(np.amax(max_currents))

    return I_Imax

def SimCSInac (C, t, p):

    C0=C[0]
    C1=C[1]
    C2=C[2]
    C3=C[3]
    C4=C[4]
    I0=C[5]
    I1=C[6]
    I2=C[7]
    I3=C[8]
    I4=C[9]
    O=C[10]


    #constants

    T = 291.0 #K or 18 degree celsius
    e =  1.602176634 * (10**-19.0) # C
    K_B = 1.380649 * (10**-23.0) # J*K^-1

    exp_factor = (e/(K_B * T)) * (10**-3)

    #Voltage sequences

    V = 0.0 #mV

    if 0 <= t < 0.10:
        V =- 90.0

    if 0.10 <= t <= p[11]:
        V = -50.0

    if p[11] < t <= 1.150:
        V = 60.0

    if t > 1.150:
        V = -90.0

    k_CI = p[8]

    k_IC = p[9]

    f = p[10]


    alpha = p[0] * np.exp(p[1] * (V * exp_factor))
    beta = p[2] * np.exp(-1.0 * p[3] * (V * exp_factor))
    k_CO = p[4] * np.exp(p[5] * (V * exp_factor))
    k_OC = p[6] * np.exp(-1.0 * p[7] * (V * exp_factor))


    # ODEs

    dC0dt = beta * C1 + (k_IC/(f**4.0)) * I0 - (k_CI*(f**4.0) + 4.0 * alpha) * C0
    dC1dt = 4.0 * alpha * C0 + 2.0 * beta * C2 + (k_IC/(f**3.0)) * I1 - (k_CI*(f**3.0) + beta + 3.0 * alpha) * C1
    dC2dt = 3.0 * alpha * C1 + 3.0 * beta * C3 + (k_IC/(f**2.0)) * I2 - (k_CI*(f**2.0) + 2.0 * beta + 2.0 * alpha) * C2
    dC3dt = 2.0 * alpha * C2 + 4.0 * beta * C4 + (k_IC/f) * I3 - (k_CI*f + 3.0 * beta + 1.0 * alpha) * C3
    dC4dt = 1.0 * alpha * C3 + k_OC * O + k_IC * I4 - (k_CI + k_CO + 4.0 * beta) * C4

    dI0dt = beta * f * I1 + (k_CI*(f**4.0)) * C0 - (k_IC/(f**4.0) + 4.0 * (alpha/f)) * I0
    dI1dt = 4.0 * (alpha/f) * I0 + 2.0 * beta * f * I2 + (k_CI*(f**3.0)) * C1 - (k_IC/(f**3.0) + beta * f + 3.0 * (alpha/f)) * I1
    dI2dt = 3.0 * (alpha/f) * I1 + 3.0 * beta * f * I3 + (k_CI*(f**2.0)) * C2 - (k_IC/(f**2.0) + 2.0 * beta * f + 2.0 * (alpha/f)) * I2
    dI3dt = 2.0 * (alpha/f) * I2 + 4.0 * beta * f * I4 + (k_CI*f) * C3 - (k_IC/f + 3.0 * beta * f + 1.0 * (alpha/f)) * I3
    dI4dt = 1.0 * (alpha/f) * I3 + k_CI * C4 - (k_IC + 4.0 * beta * f) * I4

    dOdt = k_CO * C4 - k_OC * O

    return (dC0dt, dC1dt, dC2dt, dC3dt, dC4dt, dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dOdt)

def Csi_Protocol(max_t, Deltat):
    min_t = 0.010 #mV

    t_pulse = np.linspace(min_t,max_t,np.abs(int((max_t-min_t)/Deltat))+1)

    return t_pulse #array of testing prepulses

def Csi_LS_func(x, teta):

    Vhold   =  -90.0 # mV
    Vprep   =  -50.0 # mV
    Vtest   =   60.0 # mV
    Vrepo   =  -90.0 # mV

    # conductance parameters

    #EK      = 0.0    # mV
    gK_max  = 33.2     # nS


    # Membrane capacitance
    Cm      = 1.0    # microF cm^-2

    # Time of experiments
    tini_eq   = 0    # s
    tini_prep = 0.10 # s
    time_end_pulse = 1.150
    tend = 3.00 # s

    pulse_interval = 0.030 # s
    max_pulse_interval = 0.580 # s
    min_pulse_interval = 0.010 # s

    # Time discretiztion
    dt = 1e-5

    # time array
    t = np.arange(tini_eq,tend+dt,dt)
    Npoints = len(t)

    steps = np.abs(int((max_pulse_interval-min_pulse_interval)/pulse_interval)) + 1

    Open_states = np.zeros((Npoints,steps))

    max_conductance = np.zeros(steps)

    max_currents = np.zeros(steps)

    max_conductance_prep = np.zeros(steps)

    max_currents_prep = np.zeros(steps)


    for i in range (0,steps):

        time_pulse = tini_prep + 0.010 + pulse_interval * i

        gamma = np.append(teta, time_pulse)

        f = lambda S,t: SimCSInac(S, t, gamma)

        r = odeint(f, S0, x)

        Open_states[:,i] = r[:,10]

        max_conductance[i] = gK_max * np.amax(r[int(time_pulse/dt):,10]) / np.amax(r[0:int(tini_prep/dt)+1,10])

        # Compute the current proportional to the open channel conductance and potential applied

        max_currents[i] = max_conductance[i] * (Vtest - Vprep)

    I_Imax  = (max_currents)/(np.amax(max_currents))

    return I_Imax

def SimRec (C, t, p):

    C0=C[0]
    C1=C[1]
    C2=C[2]
    C3=C[3]
    C4=C[4]
    I0=C[5]
    I1=C[6]
    I2=C[7]
    I3=C[8]
    I4=C[9]
    O=C[10]


    #constants

    T = 291.0 #K or 18 degree celsius
    e =  1.602176634 * (10**-19.0) # C
    K_B = 1.380649 * (10**-23.0) # J*K^-1

    exp_factor = (e/(K_B * T)) * (10**-3)

    #Voltage sequences

    V = 0.0 #mV

    if 0 <= t < 0.50:
        V=-90.0

    if 0.50 <= t <= 1.50:
        V=60.0

    if 1.50 < t < p[11]:
        V=-90.0

    if p[11] <= t <= 2.650:
        V=60.0

    if t > 2.650:
        V=60.0


    k_CI = p[8]

    k_IC = p[9]

    f = p[10]

    #voltage dependent rate constants

    alpha = p[0] * np.exp(p[1] * (V * exp_factor))
    beta = p[2] * np.exp(-1.0 * p[3] * (V * exp_factor))
    k_CO = p[4] * np.exp(p[5] * (V * exp_factor))
    k_OC = p[6] * np.exp(-1.0 * p[7] * (V * exp_factor))


    # ODEs

    dC0dt = beta * C1 + (k_IC/(f**4.0)) * I0 - (k_CI*(f**4.0) + 4.0 * alpha) * C0
    dC1dt = 4.0 * alpha * C0 + 2.0 * beta * C2 + (k_IC/(f**3.0)) * I1 - (k_CI*(f**3.0) + beta + 3.0 * alpha) * C1
    dC2dt = 3.0 * alpha * C1 + 3.0 * beta * C3 + (k_IC/(f**2.0)) * I2 - (k_CI*(f**2.0) + 2.0 * beta + 2.0 * alpha) * C2
    dC3dt = 2.0 * alpha * C2 + 4.0 * beta * C4 + (k_IC/f) * I3 - (k_CI*f + 3.0 * beta + 1.0 * alpha) * C3
    dC4dt = 1.0 * alpha * C3 + k_OC * O + k_IC * I4 - (k_CI + k_CO + 4.0 * beta) * C4

    dI0dt = beta * f * I1 + (k_CI*(f**4.0)) * C0 - (k_IC/(f**4.0) + 4.0 * (alpha/f)) * I0
    dI1dt = 4.0 * (alpha/f) * I0 + 2.0 * beta * f * I2 + (k_CI*(f**3.0)) * C1 - (k_IC/(f**3.0) + beta * f + 3.0 * (alpha/f)) * I1
    dI2dt = 3.0 * (alpha/f) * I1 + 3.0 * beta * f * I3 + (k_CI*(f**2.0)) * C2 - (k_IC/(f**2.0) + 2.0 * beta * f + 2.0 * (alpha/f)) * I2
    dI3dt = 2.0 * (alpha/f) * I2 + 4.0 * beta * f * I4 + (k_CI*f) * C3 - (k_IC/f + 3.0 * beta * f + 1.0 * (alpha/f)) * I3
    dI4dt = 1.0 * (alpha/f) * I3 + k_CI * C4 - (k_IC + 4.0 * beta * f) * I4

    dOdt = k_CO * C4 - k_OC * O

    return (dC0dt, dC1dt, dC2dt, dC3dt, dC4dt, dI0dt, dI1dt, dI2dt, dI3dt, dI4dt, dOdt)

def Rec_Protocol(max_t, Deltat):
    min_t = 0.0 #mV

    t_pulse = np.linspace(min_t,max_t,np.abs(int((max_t-min_t)/Deltat))+1)

    return t_pulse #array of testing prepulses

def Rec_LS_func(x, teta):

    Vhold   =  -90.0 # mV
    Vpulse  =   60.0 # mV
    Vinter  =  -90.0 # mV
    Vrep    =   60.0 # mV
    Vfin    =   60.0 # mV

    # conductance parameters

    #EK      = 0.0    # mV
    gK_max  = 33.2     # nS

    # Membrane capacitance
    Cm      = 1.0    # microF cm^-2

    # Time of experiments
    tini_eq   = 0    # s
    tini_prep = 0.50 # s
    tini_pulse = 1.50 # s

    tend = 3.00 # s

    pulse_interval = 0.030 # s
    max_pulse_interval = 0.570 # s
    min_pulse_interval = 0.000 # s

    # Time discretiztion
    dt = 1e-5

    # time array
    t = np.arange(tini_eq,tend+dt,dt)
    Npoints = len(t)

    steps = np.abs(int((max_pulse_interval-min_pulse_interval)/pulse_interval)) + 1

    Open_states = np.zeros((Npoints,steps))

    max_conductance = np.zeros(steps)

    max_currents = np.zeros(steps)

    max_conductance_prep = np.zeros(steps)

    max_currents_prep = np.zeros(steps)


    for i in range (0,steps):

        time_pulse = tini_pulse + pulse_interval * (i)

        gamma = np.append(teta, time_pulse)

        f = lambda S,t: SimRec(S, t, gamma)

        r = odeint(f, S0, x)

        Open_states[:,i] = r[:,10]

        max_conductance[i] = gK_max * np.amax(r[int(time_pulse/dt):,10])

        max_conductance_prep[i] = gK_max * np.amax(r[int(tini_prep/dt):int(tini_pulse/dt),10])

        # Compute the current proportional to the open channel conductance and potential applied

        max_currents[i] = max_conductance[i] * (Vpulse - Vhold)

        max_currents_prep[i] = max_conductance_prep[i] * (Vpulse - Vhold) # nS * mV = pA

    I_Imax = np.true_divide(max_currents,max_currents_prep)

    return I_Imax


def residual_Tot(p):

    # Time discretiztion
    tini = 0.0
    tend = 3.00
    dt = 1e-5

    t = np.arange(tini,tend+dt,dt)
    Npoints = len(t)

    ################## START OF THE 4 VOLTAGE PROTOCOLS ###################

    # ACTIVATION SEQUENCE PROTOCOL

    #here experimental data imported with Pandas in initial section
    sq_err_act = np.sum(np.subtract(y_act_WT,Act_LS_func(t,p))**2)

    act_cost_func = (1.0/len(y_act_WT)) * sq_err_act

    # INACTIVATION SEQUENCE PROTOCOL

    #here experimental data imported with Pandas in initial section
    sq_err_inact = np.sum(np.subtract(y_inact_WT,Inact_LS_func(t,p))**2)

    inact_cost_func = (1.0/len(y_inact_WT)) * sq_err_inact

    # CS INACTIVATION SEQUENCE PROTOCOL

    #here experimental data imported with Pandas in initial section
    sq_err_csi = np.sum(np.subtract(y_cs_WT,Csi_LS_func(t,p))**2)

    csi_cost_func = (1.0/len(y_cs_WT)) * sq_err_csi

    # RECOVERY SEQUENCE PROTOCOL

    #here experimental data imported with Pandas in initial section
    sq_err_rec = np.sum(np.subtract(y_rec_WT,Rec_LS_func(t,p))**2)

    rec_cost_func = (1.0/len(y_rec_WT)) * sq_err_rec

    ################## END OF THE 4 VOLTAGE PROTOCOLS ###################

    # sum of the three protocols cost function

    return (act_cost_func + inact_cost_func + csi_cost_func + rec_cost_func)

# Main

def iteration(xk,convergence):
    print('Finished iteration')
    print(xk)

# set the guess based on previous works

guess = np.array([450,0.23,2,2.2,160,0.27,245,0.33,25,0.3,0.37])

# set boundaries for parameters based on physical reasoning

boundaries = ((0.0,2000.0),(0.0,5.0),(0.0,100.0),(0.0,5.0),(0.0,1000.0),(0.0,5.0),\
              (0.0,1000.0),(0.0,5.0),(0.0,2000.0),(0.0,100.0),(0.0,1.0))

# Run local optimization for the cost function

# result = minimize(residual_Tot, guess, bounds=boundaries, method='L-BFGS-B',\
#                   options={'maxiter':15000,'maxfev':50000})

# Rub global optimization for the cost function

results = differential_evolution(residual_Tot, bounds=boundaries, maxiter=2000,\
                                 workers=-1, callback=iteration)
print (results)
