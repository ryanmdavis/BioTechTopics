## use virtual environment at /home/ryan/Dropbox/Code/Virtual-Environments/ds_py27_env
## cd /home/ryan/Dropbox/Code/DataIncubatorChallenge/BioTechTopics
## bokeh serve --show Bokeh_plot.py

from os.path import dirname, join
import os
import numpy as np
# /home/ryan/anaconda2/lib/python2.7/site-packages
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc

from BioTechTopics import Topics

#user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
#print user_paths

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=800)

## load biotech topics data
t=Topics()
t.load()
data_dict={'year': [], 'keywords': [],'TR_score':[],'tfidf_score':[],'total_score':[]}


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
source = ColumnDataSource(data=data_dict)

hover = HoverTool(tooltips=[("Who? ", "@keywords")])

p = figure(plot_height=600, plot_width=700, title="", toolbar_location=None, tools=[hover])
p.circle(x="year", y="total_score", source=source, size=7, line_color=None)



def update():
    this_query = query_term.value.strip() if bool(query_term.value.strip()) else 'antibiotic'
    t.ww2(this_query)
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "Each dot is a prominent Individual or Company related to your query" 
    source.data = t.formatSearchResults(format='tfidf_tf_product',return_top_n=int(num_results_slider.value))
    test = t.formatSearchResults(format='integrate_score',return_top_n=int(num_results_slider.value))

controls = [query_term, x_axis, y_axis, num_results_slider]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example

inputs = widgetbox(*controls, sizing_mode=sizing_mode)
l = layout([
    [desc],
    [inputs, p],
], sizing_mode=sizing_mode)

update()  # initial load of the data

curdoc().add_root(l)
curdoc().title = "BioTech Topics"
