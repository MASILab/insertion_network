from jinja2 import Environment, FileSystemLoader
from copy import deepcopy
import numpy as np

class Table:

    def __init__(
            self, 
            data,
            headers,
            caption='',
            float_format='%.2f',
            int_format='%.2f',
            template='table_core.txt',
            merge_cells=True,
            columns_to_merge=None,
            fontsize=5,
            bold=None,
            inner_text_format=None
            ):
        
        self.data = data
        self.headers = headers
        self.caption = caption
        self.float_format = float_format
        self.int_format = int_format
        self.merge_cells = merge_cells
        self.columns_to_merge = columns_to_merge
        self.fontsize = fontsize
        self.fontsize_options = ['tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge']

        template_path = '/'.join(__file__.replace("\\", '/').split('/')[: -1])
        environment = Environment(loader=FileSystemLoader(template_path + '/template'))
        self.template = environment.get_template(template)
        self.format_fns = {
            'b': lambda x: '\\textbf{%s}' % x
        }
        self.bold = bold or []
        self.inner_text_format = inner_text_format or []
    
    def escape(self, string):
        string.replace('_', r'\_')
        return string
    
    def format_datum(self):

        data_formatted = deepcopy(self.data)
        is_number = np.zeros((len(self.data), len(self.data[0])))
        for i, row in enumerate(data_formatted):
            for j, v in enumerate(row):
                
                if isinstance(v, float):
                    data_formatted[i][j] = self.float_format % v
                    is_number[i][j] = 1
                elif isinstance(v, int):
                    data_formatted[i][j] = self.int_format % v
                    is_number[i][j] = 1

                else:
                    assert isinstance(v, str)
                    if len(v) > 3 and v[0] == '<' and v[2] == '>':
                        fn_name = v[1]
                        raw_v = v[3: ].strip()
                        try:
                            raw_v = float(raw_v)
                            raw_v = self.float_format % raw_v
                            is_number[i][j] = 1
                        except ValueError:
                            pass
                        data_formatted[i][j] = self.format_fns[fn_name](raw_v)
                    elif v.find('\\\\') != -1:
                        data_formatted[i][j] = '\\makecell{%s}' % v
                if (i, j) in self.bold:
                    data_formatted[i][j] = '\\textbf{%s}' % data_formatted[i][j]
                if (i, j) in self.inner_text_format:
                    data_formatted[i][j] = self.inner_text_format[(i, j)] % data_formatted[i][j]

        return data_formatted, is_number

    def make_latex_headers(self):
        headers = self.headers
        if isinstance(headers[0], str):
            headers = [headers]

        header_strs = []
        col_format = None
        ncol = len(headers[0])
        for i, item in enumerate(headers):
            count = []
            for c in item:
                if len(count) == 0:
                    count.append([c, 1])
                else:
                    if c == count[-1][0]:
                        count[-1][1] += 1
                    else:
                        count.append([c, 1])
            row_items = []
            cur_col_format = []
            for j, (row_item, c) in enumerate(count):
                cur_col_format.append('c' * c)
                if row_item == '':
                    row_items.append(' ')
                else:
                    if j == len(count) - 1:
                        row_items.append('\\multicolumn{%d}{c}{\\textbf{%s}}' % (c, self.escape(row_item)))
                    else:
                        row_items.append('\\multicolumn{%d}{c|}{\\textbf{%s}}' % (c, self.escape(row_item)))
            header_strs.append(' & '.join(row_items))
            # use the last column names to specify the column format
            if i == len(headers) - 1:
                col_format = cur_col_format
        
        return header_strs, col_format

    def make_latex_rows(self):

        data, is_number = self.format_datum()

        nrow = len(data)
        ncol = len(data[0])

        merged_view = deepcopy(data)

        for i in range(nrow):
            for j in range(ncol):
                dij = data[i][j]
                if self.columns_to_merge is not None and j not in self.columns_to_merge:
                    continue
                if merged_view[i][j] == '' or is_number[i][j]: 
                    continue
                nr, nc = 1, 1
                ii = i + 1
                jj = j + 1
                while jj < ncol:
                    if data[i][jj] == dij:
                        nc += 1
                        merged_view[i][jj] = ''
                    else:
                        break
                    jj += 1
                jj -= 1
                while ii < nrow:
                    if data[ii][jj] == dij:
                        nr += 1
                        for k in range(j, jj + 1):
                            merged_view[ii][k] = ''
                    else:
                        break
                    ii +=1
                if nr * nc != 1:
                    merged_view[i][j] = [dij, nr, nc]

        for i in range(nrow):
            for j in range(ncol):
                # if need to merge column here
                if isinstance(merged_view[i][j], list) and merged_view[i][j][2] > 1 and merged_view[i][j][0] not in ('%', '%%'):
                    n, nr, nc = merged_view[i][j]
                    for ii in range(i, i + nr):
                        for jj in range(j, j + nc):
                            if jj == j and ii > i:
                                merged_view[ii][jj] = ['%' if ii != i + nr - 1 else '%%', nr, nc]
                            if jj > j:
                                merged_view[ii][jj] = '%'
        
        clines = []
        # the last row never needs clines
        for i, item in enumerate(merged_view[: -1]):
            cur_clines = []
            j = 0
            while j < len(item):
                v = item[j]
                # add clines for last row with merged columns
                if isinstance(v, list) and v[0] == '%%':
                    cur_clines.append('\\cline{%d-%d}' % (j + 1, j + v[2]))
                # add clines for items that are not merged
                elif isinstance(v, str) and v not in ('', '%'):
                    jj = j
                    while jj < len(item) and v not in ('', '%'):
                        jj += 1
                    cur_clines.append('\\cline{%d-%d}' % (j + 1, jj))
                    j = jj
                j += 1
            clines.append(' '.join(cur_clines))

        merged_view = [[item for item in row if item != '%'] for row in merged_view]

        row_strs = []
        if not self.merge_cells:
            merged_view = data
        for r, row in enumerate(merged_view):
            row_items = []
            for i, item in enumerate(row):
                format = 'c|' if i == 0 else '|c' if i == len(row) - 1 else '|c|'
                if isinstance(item, list):
                    n, nr, nc = item
                    if nc > 1:
                        if n != '%' and n != '%%':
                            row_items.append(
                                '\\multicolumn{%d}{%s}{\\multirow{%d}{*}{%s}}' % (nc, format, nr, n)
                            )
                        else:
                            row_items.append(
                                '\\multicolumn{%d}{%s}{}' % (nc, format)
                            )
                    else:
                        row_items.append(
                            '\\multirow{%d}{*}{%s}' % (nr, n)
                        )
                else:
                    row_items.append(item)
            row_str = ' & '.join(row_items) + ' \\\\ '
            if r < len(clines) and self.merge_cells:
                row_str += clines[r]
            row_strs.append(row_str)
        
        return row_strs

    def to_latex(self):
        
        latex_column_names, latex_column_format = self.make_latex_headers()
        latex_rows = self.make_latex_rows()

        latex = self.template.render(
            caption=self.caption,
            column_format='|'.join(latex_column_format),
            columns=latex_column_names,
            rows=latex_rows,
            fontsize=self.fontsize_options[self.fontsize]
        )

        return latex
    
    def to_latex_plt(self):

        latex = self.to_latex()
        latex_plt = latex.replace('\n', ' ')

        return latex_plt

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    plt.rc('text.latex', preamble=r'\usepackage{booktabs} \usepackage{multirow} \usepackage{caption} \usepackage{makecell}')

    data = [
        ['a', 'a', 3, 4, 5],
        ['a', 'a', 'b', 4, 5],
        ['a', 'a', 'b', 4, 'c'],
        [1, 2, 'b', '<b>4', r'this world \\ is easy']

    ]

    table = Table(
        data=data,
        caption='test table',
        headers=[['A', 'A', 'B', 'C', 'D'], ['id', 'name', 'age', 'age1', 'age2']],
        merge_cells=True
    )

    txte = table.to_latex()

    txte_plt = table.to_latex_plt()

    print(txte)

    plt.text(0.05, 0.05, txte_plt, usetex=True)
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    plt.show()



    