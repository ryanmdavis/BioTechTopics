from os.path import dirname, join
import os
import numpy as np
from bokeh.plotting import figure
from bokeh.layouts import layout, widgetbox, column
from bokeh.models import ColumnDataSource, HoverTool, Div, TapTool, OpenURL
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.io import curdoc, show

from BioTechTopics import Topics
from bokeh.embed import components 
from flask import Flask, render_template, request
import string
#test

## load biotech topics data
t=Topics()
t.load('./data/')
t.ww2('antibiotic')
data_scatter_dict = t.formatSearchResults(output_format='tfidf_tf_product',return_top_n=200)
data_line_dict = t.formatSearchResults(output_format='integrate_score')

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

## set hovertool 
hover = HoverTool(tooltips=[("Who? ", "@keywords"),("Context: ","@title")])

## Define plot p
p = figure(plot_height=400, plot_width=700, title="", toolbar_location=None, tools=[hover,"tap"])
p.circle(x="year", y="total_score", source=scatter_source, size=12, line_color=None)
p.xaxis.axis_label = "year"
p.yaxis.axis_label = "relevance"
p.title.text = "Each dot is a prominent Individual or Company related to your query"
taptool=p.select(type=TapTool)
taptool.callback=OpenURL(url="@abs_url")

# Define plot p2
p2 = figure(plot_height=200, plot_width=700, title="", toolbar_location=None)
p2.line(x="year",y="year_score",source=line_source)
p2.xaxis.axis_label="year"
p2.yaxis.axis_label="normalized term frequency"

## this function is called when user enters a new query
def update():
    this_query = query_term.value.strip().translate(None, string.punctuation) if bool(query_term.value.strip()) else 'antibiotic'
    t.ww2(this_query)
    x_name = axis_map[x_axis.value]
    y_name = axis_map[y_axis.value]

    p.xaxis.axis_label = x_axis.value
    p.yaxis.axis_label = y_axis.value
    p.title.text = "Each dot is a prominent Individual or Company related to your query :''" + this_query + "''" 
    scatter_source.data = t.formatSearchResults(output_format='tfidf_tf_product',return_top_n=int(num_results_slider.value))
    line_source.data = t.formatSearchResults(output_format='integrate_score')
    
controls = [query_term, x_axis, y_axis, num_results_slider]
for control in controls:
    control.on_change('value', lambda attr, old, new: update())

sizing_mode = 'fixed'  # 'scale_width' also looks nice with this example
inputs = widgetbox(*controls, sizing_mode=sizing_mode)
#update()  # initial load of the data

# Flask portion:
app_wwbt = Flask(__name__)
# export to a html script for embedding in Flask
#see: http://flask.pocoo.org/docs/0.12/quickstart/
@app_wwbt.route('/',methods=['GET','POST'])
def start(name=None):
    default_query='antibiotics'
    t.ww2('antibiotics')
    scatter_source.data = t.formatSearchResults(output_format='tfidf_tf_product',return_top_n=int(num_results_slider.value))
    p.title.text = "No results related to your query ''antibiotics''" 
    script, div = components(p)
    return render_template('index.html', script=script, div=div,text_value=default_query)
    
@app_wwbt.route('/index',methods=['GET','POST'])
def index(name=None):
    if request.method == 'POST':
        print('post')
        t.ww2(request.form['query'])
        scatter_source.data = t.formatSearchResults(output_format='tfidf_tf_product',return_top_n=int(num_results_slider.value))
        
        #if search results are not empty:
        if scatter_source.data['abs_url'].size > 0:
            p.title.text = "Each dot is a prominent Individual or Company related to your query ''" + request.form['query'] + "''"
            
        #if search results are empty:    
        else:
            p.title.text = "No results related to your query :''" + request.form['query'] + "''" 
    else:
        print('get')
    script, div = components(p)
    return render_template('index.html', script=script, div=div, text_value=request.form['query'])

if __name__ == "__main__":
    app_wwbt.run(port=33507)
