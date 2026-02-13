from jinja2 import Environment, FileSystemLoader
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rc('text.latex', preamble=r'\usepackage{booktabs} \usepackage{multirow}')

template_path = '/'.join(__file__.replace("\\", '/').split('/')[: -1])
environment = Environment(loader=FileSystemLoader(template_path + '/template'))
temp = environment.get_template("table_core.txt")

content = temp.render(
    caption='test',
    column_format='ccc',
    columns=['mu & sigma & Beta'],
    rows=['0 & 0.33 & 2.099', '0 & 1.00 & 2.359']
    )

txte = content.replace('\n', ' ')

print(txte)

plt.text(0.1, 0.4, txte, fontsize=14, usetex=True)
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

plt.show()

print(1)