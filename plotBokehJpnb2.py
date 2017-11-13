## use virtual environment at /home/ryan/Dropbox/Code/Virtual-Environments/ds_py27_env
## cd /home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics
## bokeh serve --show Bokeh_plot.py

from os.path import dirname, join
import os
import numpy as np
# /home/ryan/anaconda2/lib/python2.7/site-packages
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, gridplot, column
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc, output_notebook, show
from BioTechTopics import Topics

def plotBokehInJpnb(t,query_term):
    output_notebook()

    #user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    #print user_paths
    
    desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)
    
    ## load biotech topics data
    #t=Topics()
    #t.load()
    t.ww2(query_term)
    data_scatter_dict = t.formatSearchResults(format='tfidf_tf_product',return_top_n=200)
    data_line_dict = t.formatSearchResults(format='integrate_score')
    
    ## define axis titles
    axis_map = {
        "Relevance": "Relevance",
        "Year": "Year",
    }
    
    
    ## Create Input controls
    query_term = TextInput(title="Search",value="antibiotics")
    num_results_slider = Slider(start=10, end=400, value=200, step=10, title="Number of Results")
    x_axis = Select(title="X Axis", options=["Year"], value="Year")
    y_axis = Select(title="Y Axis", options=["Relevance"], value="Relevance")
    
    # Create Column Data Source that will be used by the plot
    scatter_source = ColumnDataSource(data=data_scatter_dict)
    line_source = ColumnDataSource(data=data_line_dict)
    
    hover = HoverTool(tooltips=[("Who? ", "@keywords")])
    
    p = figure(plot_height=400, plot_width=700, title="", toolbar_location=None, tools=[hover])
    p.circle(x="year", y="total_score", source=scatter_source, size=7, line_color=None)
    p.xaxis.axis_label = "year"
    p.yaxis.axis_label = "relevance"
    p.title.text = "Each dot is a prominent Individual or Company related to your query"
    
    p2 = figure(plot_height=200, plot_width=700, title="", toolbar_location=None)
    p2.line(x="year",y="normalized_year_score",source=line_source)
    p2.xaxis.axis_label="year"
    p2.yaxis.axis_label="normalized term frequency"
    
    show(column(p,p2))
    
   
#     controls = [query_term, x_axis, y_axis, num_results_slider]
#     for control in controls:
#         control.on_change('value', lambda attr, old, new: update())
#     
#     sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
#     
#     inputs = widgetbox(*controls, sizing_mode=sizing_mode)
#     
#     #under layout, switched "p" and "grid"
#     l = layout([
#         [desc],
#         [inputs, grid],
#     ], sizing_mode=sizing_mode)
#     
#     update()  # initial load of the data
#     
#     curdoc().add_root(l)
#     curdoc().title = "BioTech Topics"
