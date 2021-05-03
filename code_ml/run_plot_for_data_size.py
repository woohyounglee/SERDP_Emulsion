import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# sheet_name = 'Classification'
sheet_name = 'Regression'

score = 'Averaged F-1 score'

if sheet_name == 'Regression':
    score = 'Averaged MAE'

o_df = pd.read_excel(open('../output_for_paper/[05-02-2021] Learning_curve.xlsx', 'rb'),
                     sheet_name=sheet_name, index_col=0, header=None)

plt.plot(o_df[1], 'r*-')
plt.plot(o_df[2], 'bs-')
plt.plot(o_df[3], 'g^-')

plt.legend(["OV (Oily value)", "OS (Oil separation)", "Turbidity"])

plt.title(sheet_name)
plt.ylabel(score)
plt.xlabel('# of data')
plt.show()

