

original_questions = {
     #Format is 'question':[options]
     'Fixed surfactant information':{
          'Critical micelle concentration (CMClog) (ppm)': ['87 ~ 3824'],
          'Equilibrium surface tension (ST) above CMC (air) (mN/M)': ['15.2 ~ 35.4'],
          'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': ['1.3 ~ 14.6'],
          'Micelle size (nm)': ['1.3 ~ 111'],
          'Zeta potential (mV)': ['-63 ~ -13'],
          'Alkalinity (mg CaCO3/L)': ['0 ~ 6014'],
          "Surfactant's Initial pH at 7CMC": ['6.3 ~ 12.2'],
     },
     'Editable variables':{
          'Surfactant concentration (ppm)': ['0 ~ 26768'],
          'pH': ['4 ~ 12.2'],
          'Suspended solids concentration (ppm)': ['0 ~ 2000'],
          'Salinity (ppm)': ['0 ~ 35000'],
          'Temperature (°C)': ['4 ~ 35'],
     }
}


type_questions = {
     'Surfactant name': 'combobox',
     'Surfactant concentration (ppm)': 'editbox',
     'Critical micelle concentration (CMClog) (ppm)': 'editbox',
     'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 'editbox',
     'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 'editbox',
     'Micelle size (nm)': 'editbox',
     'Zeta potential (mV)': 'editbox',
     'Alkalinity (mg CaCO3/L)': 'editbox',
     "Surfactant's Initial pH at 7CMC": 'editbox',
     'pH': 'editbox',
     'Suspended solids concentration (ppm)': 'editbox',
     'Salinity (ppm)': 'editbox',
     'Temperature (°C)': 'editbox',
}

enable_questions = {
     'Surfactant name': 'true',
     'Surfactant concentration (ppm)': 'true',
     'Critical micelle concentration (CMClog) (ppm)': 'false',
     'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 'false',
     'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 'false',
     'Micelle size (nm)': 'false',
     'Zeta potential (mV)': 'false',
     'Alkalinity (mg CaCO3/L)': 'false',
     "Surfactant's Initial pH at 7CMC": 'false',
     'pH': 'true',
     'Suspended solids concentration (ppm)': 'true',
     'Salinity (ppm)': 'true',
     'Temperature (°C)': 'true',
}

default_values = {
     # ['AFFF','B&B','Blast','Calla','Powergreen','PRC','SDS','Surge',questions_type'Triton-X-100','Type 1']
     'None': {
          'Critical micelle concentration (CMClog) (ppm)': 0,
          'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 0,
          'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 0,
          'Micelle size (nm)': 0,
          'Zeta potential (mV)': 0,
          'Alkalinity (mg CaCO3/L)': 0,
          "Surfactant's Initial pH at 7CMC": 0,
     },
    'AFFF': {
          'Critical micelle concentration (CMClog) (ppm)': 3399,
          'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 15.2,
          'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 2.3,
          'Micelle size (nm)': 55,
          'Zeta potential (mV)': -24,
          'Alkalinity (mg CaCO3/L)': 68.9,
          "Surfactant's Initial pH at 7CMC": 6.8,
     },
     'B&B': {
               'Critical micelle concentration (CMClog) (ppm)': 361,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 25,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 7.8,
               'Micelle size (nm)': 21,
               'Zeta potential (mV)': -28,
               'Alkalinity (mg CaCO3/L)': 21,
               "Surfactant's Initial pH at 7CMC": 8.8,
          },
     'Blast': {
               'Critical micelle concentration (CMClog) (ppm)': 934,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 27.7,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 14.6,
               'Micelle size (nm)': 111,
               'Zeta potential (mV)': -63,
               'Alkalinity (mg CaCO3/L)': 171,
               "Surfactant's Initial pH at 7CMC": 10.5,
          },
     'Calla': {
               'Critical micelle concentration (CMClog) (ppm)': 328,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 27.3,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 5.7,
               'Micelle size (nm)': 5,
               'Zeta potential (mV)': -39,
               'Alkalinity (mg CaCO3/L)': 27.1,
               "Surfactant's Initial pH at 7CMC": 8.6,
          },
     'Powergreen': {
               'Critical micelle concentration (CMClog) (ppm)': 3824,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 26.1,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 3.5,
               'Micelle size (nm)': 7,
               'Zeta potential (mV)': -13,
               'Alkalinity (mg CaCO3/L)': 366.5,
               "Surfactant's Initial pH at 7CMC": 9.9,
          },
     'PRC': {
               'Critical micelle concentration (CMClog) (ppm)': 1871,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 30.5,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 1.3,
               'Micelle size (nm)': 6,
               'Zeta potential (mV)': -13,
               'Alkalinity (mg CaCO3/L)': 6.4,
               "Surfactant's Initial pH at 7CMC": 6.9,
          },
     'SDS': {
               'Critical micelle concentration (CMClog) (ppm)': 1983,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 35.4,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 2.75,
               'Micelle size (nm)': 1.3,
               'Zeta potential (mV)': -40,
               'Alkalinity (mg CaCO3/L)': 7.4,
               "Surfactant's Initial pH at 7CMC": 6.3,
          },
     'Surge': {
               'Critical micelle concentration (CMClog) (ppm)': 98,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 27.4,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 7.3,
               'Micelle size (nm)': 23,
               'Zeta potential (mV)': -15,
               'Alkalinity (mg CaCO3/L)': 412.5,
               "Surfactant's Initial pH at 7CMC": 12.2,
          },
     'Triton-X-100': {
               'Critical micelle concentration (CMClog) (ppm)': 102,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 29.6,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 2.1,
               'Micelle size (nm)': 9.1,
               'Zeta potential (mV)': -16,
               'Alkalinity (mg CaCO3/L)': 3.5,
               "Surfactant's Initial pH at 7CMC": 6.7,
          },
     'Type 1': {
               'Critical micelle concentration (CMClog) (ppm)': 87,
               'Equilibrium surface tension (ST) above CMC (air) (mN/M)': 30.2,
               'Equilibrium interfacial tension (IFT) above CMC with NSBM (mN/M)': 2.5,
               'Micelle size (nm)': 18,
               'Zeta potential (mV)': -28,
               'Alkalinity (mg CaCO3/L)': 3.3,
               "Surfactant's Initial pH at 7CMC": 6.4,
          }
}