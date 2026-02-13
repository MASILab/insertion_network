import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import json
from copy import deepcopy
from .latex_table import Table
from scipy.stats import wilcoxon
from matplotlib import colors

class Chart:
    
    color_scheme1 = ['#fb8072', '#80b1d3', '#b3de69', '#fdb462', '#8dd3c7', '#ffffb3', '#bebada', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', 'ffed6f']
    color_scheme2 = ['#D7191C', '#2C7BB6', '#00FF00', 'orange', '#a6a832', '#3632a8', '#D7191C', '#2C7BB6', "purple", '#bc80bd', '#ccebc5', '#ffed6f']

    legend_fontsize = 10
    legend_loc = 'upper right'
    legend_ncol = 1
    innertext_fontsize = 12
    xticks_fontszie = 12
    yticks_fontszie = 12
    xlabel_fontsize = 12
    ylabel_fontsize = 12
    title_fontsize = 12
    figwidth = 4
    figheight = 4

    def __init__(self, fig=None) -> None:
        if fig is None:
            fig = plt.figure(figsize=(self.figwidth, self.figheight))
        self.fig = fig
        self.ax = fig.add_subplot()
        self.ax.tick_params('x', labelsize=self.xticks_fontszie)
        self.ax.tick_params('y', labelsize=self.yticks_fontszie)
        self.colors = self.color_scheme1
    
    def show(self):
        plt.show()
    
    def config(self, xticks_fontsize, yticks_fontsize):
        self.ax.tick_params('x', labelsize=xticks_fontsize)
        self.ax.tick_params('y', labelsize=yticks_fontsize)
    
    def title(self, title, fontsize=None, y=None, pad=None):
        self.ax.set_title(title, fontsize=fontsize or self.title_fontsize, y=y, pad=pad)
        return self

    def set_legend_loc(self, loc):
        self.legend_loc = loc
    
    def set_colors(self, colors):
        self.colors = colors
    
    def set_figsize(self, width, height):
        self.fig.set_figwidth(width)
        self.fig.set_figheight(height)
        return self
    
    def table(self, data, headers, caption='', float_format='%.2f', 
              int_format='%.2f', template='table_core.txt', merge_cells=True, columns_to_merge=None, fontsize=5, bold=None, inner_text_format=None, print_latex=False):
        latex_table = Table(data, headers, caption, float_format, int_format, template, merge_cells, columns_to_merge, fontsize, bold=bold, inner_text_format=inner_text_format)
        latex_plt = latex_table.to_latex_plt()
        if print_latex:
            print(latex_table.to_latex())
        plt.rc('text.latex', preamble=r'\usepackage{booktabs} \usepackage{multirow} \usepackage{caption} \usepackage{makecell} \usepackage{xcolor}')
        self.ax.set_axis_off()
        self.ax.text(0.05, 0.05, latex_plt, usetex=True)
    
    def show_legend(self, handlers=None, labels=None, frameon=True, markerfirst=True):
        if handlers is not None:
            self.ax.legend(handlers, labels, loc=self.legend_loc, fontsize=self.legend_fontsize, 
                           frameon=frameon, ncol=self.legend_ncol, markerfirst=markerfirst)
        else:
            self.ax.legend(loc=self.legend_loc, fontsize=self.legend_fontsize, frameon=frameon, ncol=self.legend_ncol, markerfirst=markerfirst)

    def line(self, x, y, color=0, opacity=1, linewidth=None, label=None, linestyle='-', zorder=0):
        if isinstance(color, int):
            color = self.colors[color]
        self.ax.plot(x, y, color=color, alpha=opacity, linewidth=linewidth, label=label, linestyle=linestyle, zorder=zorder)
    
    def axis_off(self):
        self.ax.set_axis_off()
        return self

    def frame_off(self):
        self.ax.set_frame_on(False)
        return self

    def xaxis_off(self):
        self.ax.get_xaxis().set_visible(False)
        return self
    
    def yaxis_off(self):
        self.ax.get_yaxis().set_visible(False)
        return self
    

    def text(self, x, y, text, color=None, font_size=10):
        self.ax.text(x, y, s=text, color=color, fontdict={'size': font_size})

    def scatter(self, x, y, color, label=None, marker='.', markersize=None, linewidths=1, opacity=1, zorder=0):

        if markersize is not None:
            markersize = plt.rcParams['lines.markersize'] ** 2 * markersize
            
        if isinstance(color, int):
            color = self.colors[color]
        self.ax.scatter(x, y, color=color, marker=marker, s=markersize, alpha=opacity, label=label,linewidths=linewidths, zorder=zorder)
    
    def slice(self, volume, orientation, slice_index='mid', voxsz=(1, 1, 1), cmap='gray', origin='lower', mask_overlays=None, contour_overlays=None, contour_overlay_linewidth=None):
        
        def get_im2d(im3d, orientation, slice_index):
            if orientation == 'x':
                if slice_index == 'mid':
                    slice_index = im3d.shape[0] // 2
                slice = im3d[slice_index, :, :]
                aspect = voxsz[2] / voxsz[1]
            
            if orientation == 'y':
                if slice_index == 'mid':
                    slice_index = im3d.shape[1] // 2
                slice = im3d[:, slice_index, :]
                aspect = voxsz[2] / voxsz[0]
            
            if orientation == 'z':
                if slice_index == 'mid':
                    slice_index = im3d.shape[2] // 2
                slice = im3d[:, :, slice_index]
                aspect = voxsz[1] / voxsz[0]
            return slice, aspect
        
        im2d, aspect = get_im2d(volume, orientation, slice_index)

        self.ax.imshow(im2d.T, cmap=cmap, origin=origin)

        if mask_overlays is not None:
            for item in mask_overlays:
                if not isinstance(item, list):
                    mask_data, color, opacity = item, 'red', 0.5
                if len(item) == 1:
                    mask_data, color, opacity = item[0], 'red', 0.5
                if len(item) == 2:
                    mask_data, color, opacity = item[0], item[1], 0.5
                if len(item) == 3:
                    mask_data, color, opacity = item

                mask2d, _ = get_im2d(mask_data, orientation, slice_index)
                ones = np.argwhere(mask2d == 1)
                if ones.shape[0] > 0:
                    # self.ax.plot(ones[:, 0], ones[:, 1], color=color, alpha=1, marker=',', lw=1, linestyle="")
                    self.ax.contourf(mask2d.T, levels=[0.5, 1], colors=color, alpha=opacity)
        
        if contour_overlays is not None:
            for item in contour_overlays:
                if not isinstance(item, list):
                    mask_data, color, opacity = item, 'red', 0.5
                if len(item) == 1:
                    mask_data, color, opacity = item[0], 'red', 0.5
                if len(item) == 2:
                    mask_data, color, opacity = item[0], item[1], 0.5
                if len(item) == 3:
                    mask_data, color, opacity = item
                mask2d, _ = get_im2d(mask_data, orientation, slice_index)
                ones = np.argwhere(mask2d == 1)
                if ones.shape[0] > 0:
                    # self.ax.plot(ones[:, 0], ones[:, 1], color=color, alpha=1, marker=',', lw=1, linestyle="")
                    self.ax.contour(mask2d.T, levels=[0.5, 1], colors=color, alpha=opacity, linewidths=contour_overlay_linewidth)

        self.ax.set_aspect(aspect=aspect)

    def gray_image(self, image, pixsz=(1, 1), origin='lower'):
        aspect = pixsz[1] / pixsz[0]
        self.ax.imshow(image.T, cmap='gray', aspect=aspect, origin=origin)
    
    def bar(self, x, y, color='red', label='', show_values=False, value_y_offset=0.01, value_format='%.2f', width=1):
        self.ax.bar(x, y, color=color, label=label, width=width)
        if show_values:
            for index, value in enumerate(y):
                self.ax.text(index, value + value_y_offset, value_format % value, ha='center', fontsize=self.innertext_fontsize)

    def boxplot(self, data, bar_labels=None, series_labels=None, showfliers=False, 
                show_stats='median', 
                significance_pairs=None,
                significance_test_use_color=False,
                significance_test_method='wilcoxon',
                significance_valid_percentile=0,
                significance_text_x_offest=0,
                significance_text_y_offest=0.5,
                significance_bar_top_offset=4,
                significance_bar_verticle_line_length_ratio=0.1,
                barwidth=0.4, space_between_bars=0.2, space_between_groups=1,
                auto_figsize=True, style='2', stats_format='%.2f',
                stats_label_x_offset=0,
                stats_label_y_offset=0.05,
                legend_frameon=True
                ):

        def set_box_color_style1(bp, color):
            plt.setp(bp['boxes'], color=color)
            plt.setp(bp['whiskers'], color=color)
            plt.setp(bp['caps'], color=color)
            plt.setp(bp['medians'], color=color)
        
        def set_box_color_style2(bp, color):
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                plt.setp(bp['medians'], color='black')
            

        num_series = len(data)
        num_bars = len(data[0])

        if auto_figsize and isinstance(self.fig, plt.Figure):
            height = self.fig.get_figheight()
            width = 0.6 * num_series * num_bars
            self.set_figsize(width=width, height=height)

        first_series_bar_positions = np.array(range(num_bars)) * num_series
        first_series_bar_positions = np.array([i * ((barwidth + space_between_bars) * (num_series - 1) + space_between_groups) for i in range(num_bars)])

        bar_topcoords = []
        legend_marker = []

        for i, item in enumerate(data):
            bar_positions = first_series_bar_positions + (barwidth + space_between_bars) * i
            boxes = self.ax.boxplot(
                item,
                positions=bar_positions,
                patch_artist=style == '2',
                sym='.',
                widths=barwidth,
                showfliers=showfliers
            )

            style_funcs = {'1': set_box_color_style1, '2': set_box_color_style2}
            style_funcs[style](boxes, self.colors[i])
            legend_marker.append(boxes['boxes'][0])

            cur_bar_topcoords = []
            for j in range(num_bars):
                x, y = boxes['whiskers'][j * 2 + 1].get_xydata().max(axis=0)
                if showfliers:
                    fliers = boxes['fliers'][j].get_xydata()
                    if fliers.shape[0] > 0:
                        x, y = fliers.max(axis=0)
                cur_bar_topcoords.append([x, y])
            
            bar_topcoords.append(cur_bar_topcoords)

        data_max, data_min = 0, 0
        for item in data:
            for v in item:
                data_max = max(data_max, max(v))
                data_min = min(data_min, min(v))

        if show_stats is not None:
            for i in range(len(data)):
                for j in range(num_bars):
                    x, y = bar_topcoords[i][j]
                    cur_data = data[i][j]
                    if show_stats == 'median':
                        stats = np.median(cur_data)
                    elif show_stats == 'mean':
                        stats = np.mean(cur_data)
                    else:
                        stats = 0
                    if np.std(cur_data) != 0:
                        self.ax.text(x - stats_label_x_offset, y + stats_label_y_offset * np.mean(data[0][0]), stats_format % stats, color='black', fontsize=self.innertext_fontsize, horizontalalignment='center')

        if significance_pairs is not None:

            verticle_gap_ratio = 0.03
            verticle_line_ratio = significance_bar_verticle_line_length_ratio

            left_bar_topcoords = deepcopy(bar_topcoords)
            right_bar_topcoords = deepcopy(bar_topcoords)
            
            ymin = self.ax.dataLim.get_points()[0, 1]
            ymax = self.ax.dataLim.get_points()[1, 1]

            new_ymax = ymax

            for pair in significance_pairs:

                x1 = bar_topcoords[pair[0][1]][pair[0][0]][0]
                x2 = bar_topcoords[pair[1][1]][pair[1][0]][0]

                d1 = data[pair[0][1]][pair[0][0]]
                d2 = data[pair[1][1]][pair[1][0]]

                if significance_test_method == 'wilcoxon':
                    s, p_value = wilcoxon(d1, d2)
                if significance_test_method == 't-test':
                    valid_d1 = np.array(d1)
                    valid_d1 = np.sort(valid_d1)
                    i1 = int(len(valid_d1) * significance_valid_percentile)
                    if i1 > 0:
                        valid_d1 = valid_d1[i1: -i1]

                    valid_d2 = np.array(d2)
                    valid_d2 = np.sort(valid_d2)
                    i2 = int(len(valid_d2) * significance_valid_percentile)
                    if i2 > 0:
                        valid_d2 = valid_d2[i2: -i2]
                    p_value = two_sample_t_test(valid_d1, valid_d2)

                if p_value > 0.05:
                    sig = 'ns'
                elif 0.01 < p_value <= 0.05:
                    sig = '*'
                elif 0.001 < p_value <= 0.01:
                    sig = '**'
                else:
                    sig = '***'
                
                print(f'{series_labels[pair[0][1]]}({bar_labels[pair[0][0]]}) v.s {series_labels[pair[1][1]]}({bar_labels[pair[1][0]]}): ', p_value, sig)

                delta_x = 0.1 * barwidth
                if x1 < x2:
                    bar_topcoords1 = right_bar_topcoords
                    bar_topcoords2 = left_bar_topcoords
                    x1 += delta_x
                    x2 -= delta_x
                else:
                    bar_topcoords1 = left_bar_topcoords
                    bar_topcoords2 = right_bar_topcoords
                    x1 -= delta_x
                    x2 += delta_x
                
                y1 = bar_topcoords1[pair[0][1]][pair[0][0]][1]
                y2 = bar_topcoords2[pair[1][1]][pair[1][0]][1]
                yx = [bar_topcoords2[j][pair[1][0]][1] for j in range(pair[0][1], pair[1][1] + 1)]

                y = max(yx) + verticle_line_ratio * ymax + significance_bar_top_offset
                y1 += verticle_gap_ratio * ymax + significance_bar_top_offset
                y2 += verticle_gap_ratio * ymax + significance_bar_top_offset
                
                self.line([x1, x1], [y1, y], color='black')
                self.line([x2, x2], [y2, y], color='black')
                self.line([x1, x2], [y, y], color='black')

                color = 'black'
                if significance_test_use_color:
                    if sig == 'ns':
                        color = 'black'
                    else:
                        if np.mean(d1) > np.mean(d2):
                            color=significance_test_use_color[0]
                        else:
                            color=significance_test_use_color[1]
                self.ax.text((x1 + x2) / 2 + significance_text_x_offest, y + significance_text_y_offest, sig, color=color, fontsize=self.innertext_fontsize, horizontalalignment='center')

                bar_topcoords1[pair[0][1]][pair[0][0]][1] = y
                bar_topcoords2[pair[1][1]][pair[1][0]][1] = y
                new_ymax = max(y + verticle_gap_ratio * ymax, new_ymax)
            
            self.yticks(minv=ymin, maxv=new_ymax)

                
        if show_stats is not None:
            stats = []
        
        if series_labels is not None:
            self.ax.legend(legend_marker, series_labels, loc=self.legend_loc, fontsize=self.legend_fontsize, ncol=self.legend_ncol, frameon=legend_frameon)
        
        if num_series % 2 == 0:
            xtics = first_series_bar_positions + (barwidth + space_between_bars) * (num_series / 2) - (barwidth + space_between_bars) / 2
        else:
            xtics = first_series_bar_positions + (barwidth + space_between_bars) * (num_series // 2)

        if bar_labels is None:
            bar_labels = ['%.2f' % p for p in xtics]
        self.xticks(ticks=xtics, labels=bar_labels)
    
    def xlabel(self, label):
        self.ax.set_xlabel(label, fontsize=self.xlabel_fontsize)
        return self
    
    def ylabel(self, label):
        self.ax.set_ylabel(label, fontsize=self.ylabel_fontsize)
        return self

    def xticks(self, minv=None, maxv=None, num=None, ticks=None, labels=None, rotation=None):
        if ticks is None and num is None:
            self.ax.set_xlim(left=minv, right=maxv)
            return self
        if ticks is None:
            ticks = np.linspace(start=minv, stop=maxv, num=num)
        self.ax.set_xlim(left=minv, right=maxv)
        if labels is None:
            labels = ticks
        self.ax.set_xticks(ticks, labels=labels, rotation=rotation)
        if rotation is not None:
            self.ax.tick_params(axis='x', rotation=rotation)
        return self
    
    def yticks(self, minv=None, maxv=None, num=None, ticks=None, labels=None):
        if ticks is None and num is None:
            self.ax.set_ylim(bottom=minv, top=maxv)
            return self
        if ticks is None:
            ticks = np.linspace(start=minv, stop=maxv, num=num)
        self.ax.set_ylim(top=maxv, bottom=minv)
        self.ax.set_yticks(ticks, labels=labels)
        return self

class ChartGroup:

    chart_cls = Chart

    def __init__(self, nrows, ncols, each_width=None, each_height=None, width_ratios=None, height_ratios=None) -> None:
        self.fig = plt.figure(constrained_layout=True)

        if width_ratios is None:
            width_ratios = [1] * ncols
        if height_ratios is None:
            height_ratios = [1] * nrows

        subfigs = self.fig.subfigures(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)

        if each_height is None:
            each_height = self.chart_cls.figheight
        if each_width is None:
            each_width = self.chart_cls.figwidth

        total_height = sum(height_ratios) * each_height * 0.9
        total_width = sum(width_ratios) * each_width * 0.9

        self.fig.set_figwidth(total_width)
        self.fig.set_figheight(total_height)

        self.charts = []
        x = 0
        for i  in range(nrows):
            charts_row = []
            for j in range(ncols):
                if nrows == 1 or ncols == 1:
                    charts_row.append(self.chart_cls(subfigs[x]))
                else:
                    charts_row.append(self.chart_cls(subfigs[i, j]))
                x += 1
            self.charts.append(charts_row)
    
    def get_chart(self, row, col) -> Chart:
        return self.charts[row - 1][col - 1]
    
    def show(self):
        plt.show()