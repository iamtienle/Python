# Importing required libraries

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import *
from collections import Counter
import pydeck as pdk

# -------------------------------------------------------------- #
#  Sidebar filters are written on upper section of code so we can
#  access their variables in lower section and update the right panel.
# -------------------------------------------------------------- #


# Configuring the Streamlit UI
plt.rcParams['font.size'] = 5.0
st.set_page_config(layout="wide", page_title='UFO sightings', page_icon='🛸')
styles_sheet = open("style.css", 'r', encoding="utf-8").read()
st.markdown(f"<style> {styles_sheet} </style>", unsafe_allow_html=True, )

# importing csv file
main_df = None


def readCSVFile():
    global main_df
    main_df = pd.read_csv("data.csv", error_bad_lines=False)
    # main_df = pd.read_csv("data.csv",low_memory=False)


readCSVFile()

# Sorting df by country
main_df = main_df.sort_values(by=['country'], ascending=True)


# Some functions to extract/manipulate data from csv file. linked with filters
def generateDateTimeValue(x, month=False):
    index = -1 if not month else -3
    if '/' in str(x):
        # In a datetime string we split the string first with " " to get the date only without time and then again split the
        # string with "/" to get the year or month. index = -1 gives us the year while index = -3 gives us the month
        return int(str(x).split(" ")[0].split("/")[index])
    else:
        if index == -3:
            return datetime.fromtimestamp(int(x)).month
        return datetime.fromtimestamp(int(x)).year


def convertToFloat(x):
    # """There are 2 datatypes in the longitude and latitude columns.The string and the float.
    # Strings can be normal sentences, dates as well as floating point values but readed as string.
    # So we checked the datatype first. If it is string then we check whether it's convertable to a
    # float or not. This is done by first checking date condition ( '/' not in x ) as date contains a '/'.
    # Then by checking every character of x not an alphabet( any used to check the presence of any alphabetic
    # character in x). If the conditions are satisfied then the string is convertible in a float else a 0 is
    # returned. If the type is a float then we can directly return x."""

    if type(x) == str:
        if '/' not in x and not any(c.isalpha() for c in x):
            return float(x)
        else:
            return 0
    elif type(x) == float:
        return x


# """Returns 3 things. 1) All the countries(the order of the country in this list matters as it shows how many
# times a shape has been in that country. 2) shape_container : shows that a particular shape has been in a country
# for how many times. for example if our countries array is  ['ca', 'gb', 'us', 'de', 'au'] and our shape_container is
# [{'disk': [7, 0, 124, 0, 0]}, {'circle': [1, 2, 41, 0, 0]}, {'light': [0, 0, 39, 0, 0]}, {'cigar': [2, 1, 34, 0, 0]}]
# then this shows that disk shape has been seen in ca for 7 times, in gb for 0 times, in USA for 124 times and so on..
# 3) The map_df containing lon lat info for mapping on the world map. First we start this fn by filtering the df for
# particular interval and setting up the map_df.  We then filtered out our dataframe for not null values of the countries.
# ufo_shapes is a list containing all the shapes for the not null countries. The counter in the next line returns a dictionary
# with the count of each shape. Finally we select the 4 most common ufo_shapes. Now after getting the most common UFO_shapes we
# filter again our dataframe and countries with the one containing those shapes.
# )"""
def getStackedBarChartData(year_slide_range):
    df = main_df[(main_df.year >= year_slide_range[0]) & (main_df.year <= year_slide_range[-1])]
    # print(list(coun))
    # print(list(coun).remove('nan'))
    # coun = list(coun).remove('nan')

    # print(coun)
    map_df = pd.DataFrame(
        [],
        columns=['lat', 'lon', 'shape']
    )

    # Commented below code here and added to the last of this function so that we get our filtered df of only top 4 shapes
    # map_df['lat'] = df['latitude'].values.tolist()
    # map_df['lon'] = df['longitude'].values.tolist()
    # map_df['country'] = df['country'].values.tolist()

    countries = [x for x in list(set(df['country'].values.tolist())) if str(x) != 'nan']
    df = df[(df["country"].notnull()) & (df["country"] != u"")]
    ufo_shapes = [x for x in list((df['shape'].values.tolist())) if str(x) != 'nan']
    c = Counter(ufo_shapes)
    ufo_shapes = list(c.most_common(4))
    ufo_shapes = [shape[0] for shape in ufo_shapes]
    df = df[df['shape'].str.contains("|".join(ufo_shapes), na=False)]
    countries = [x for x in list(set(df['country'].values.tolist())) if str(x) != 'nan' and str(x) != 'NaN']
    data_container = []

    # Now in the for loop we populate our shape_container by iterating over the shapes
    # in the ufo_shapes and getting each shape's country's count. data_container will be a
    # list of dictionaries having structure [{},{},{},{}]. In each dictionary there is only 1
    # item whose key is the shape name and the value will be a list of each country's count.
    # A dictionary will look something like this : {'triangle':[23,0,4,5,1]}. Where 23 is the no
    # of times this shape UFO has been spotted in a countries[0] ...
    for shape in ufo_shapes:
        shape_container = []
        for country in countries:
            record = df[(df['shape'].str.contains(shape)) & (df['country'].str.contains(country))].shape[0]
            shape_container.append(record)
        data_container.append({shape: shape_container})

    # our data_container(shape_container) shouldn't have less than 4 shapes.
    while len(data_container) < 4:
        data_container.append({'': [0]})

    # For legends, getting all countries in the coun array from main_df and removing nan
    coun = main_df.country.unique()
    coun = list(coun)
    coun = [x for x in coun if str(x) != 'nan']
    # Appending those countries not present in the filtered dataframe's country columns
    for c in coun:
        if c not in countries:
            countries.append(c)

    # Adjusting data_container according to the appended countries
    for data in data_container:
        for key, item in data.items():
            while len(item) < 5:
                data[key].append(0)

    map_df['lat'] = df['latitude'].values.tolist()
    map_df['lon'] = df['longitude'].values.tolist()

    return {
        'countries': countries,
        'shape_container': data_container,
        'map_df': map_df,
        # 'map_sub_df':map_sub_df,
        # 'shapes':shapes
    }


# Reshaping dataframe for required functionality -  making more reliable
# Adding year and months column in our dataframe to get the year and months more easily, otherwise we would have
# been extracting the year and month every time when needed.
main_df['year'] = main_df.apply(lambda x: generateDateTimeValue(x['datetime']), axis=1)
main_df['month'] = main_df.apply(lambda x: generateDateTimeValue(x['datetime'], True), axis=1)
# dropping the rows in our dataframe whose latitude or longitude value is null
main_df = main_df[main_df['latitude'].notna()]
main_df = main_df[main_df['longitude'].notna()]
# longitude and latitude values currently can be of string as well as float datatype. So converting them to float only
main_df['longitude'] = main_df.apply(lambda x: convertToFloat(x['longitude']), axis=1)
main_df['latitude'] = main_df.apply(lambda x: convertToFloat(x['latitude']), axis=1)


# Function to extract data for pie chart - linked with hemisphere(s) filter too
def getPieChartData(hemispheres):
    df = main_df

    # """If len(hemisphere) == 1 shows that an option has been selected from the Choose hemisphere options. A hemisphere can be
    # northern or southern. So if no option or both options is/are selected then our dataframe should contain the data for both hemispheres.
    # However for a certain option like for example Southern then the dataframe should be filtered out and that is exactly happening
    # here. res is a dictionary containing the count for each season. Each season is get by filtering the dataframe according to the
    # conditions like if month is >2 and <6 then it is a spring etc. The .shape[0] in the end of each res line indicates the no of rows
    # satisfying the condition of months."""

    if len(hemispheres) == 1:
        if hemispheres[0] == 'Northern':
            df = df[(df.latitude > 0)]
        else:
            df = df[(df.latitude < 0)]

    res = {
        "Spring": df[(df.month > 2) & (df.month < 6)].shape[0],
        "Summer": df[(df.month > 5) & (df.month < 9)].shape[0],
        "Fall": df[(df.month > 8) & (df.month < 12)].shape[0],
        "Winter": df[(df.month > 11) | (df.month < 3)].shape[0],
    }
    # print(res)
    return res


# ---------------------------------------- Start: Sidebar -----------------------------------

# Hemisphere filter - Point 1 filter
hemisphere = ('Northern', 'Southern',)
hemisphere_multi_select = st.sidebar.multiselect(
    'Please select hemisphere(s)',
    hemisphere
)

# Year range slider - Point 2 filter
available_years = list(dict.fromkeys(main_df['year']))
available_years.sort()
year_slide_range = st.sidebar.select_slider('Please select year Range', options=available_years,
                                            value=(available_years[0], available_years[-1]))

# Radio buttons - Point 3 filter
country_vs_state_option = st.sidebar.radio(
    'Please select an option:',
    (
        "Countries",
        "States",
    )
)

st.sidebar.markdown(f"<span>You have selected  <b>  <u> {country_vs_state_option} </u></b></span>",
                    unsafe_allow_html=True)
st.sidebar.markdown(f"<span>Please select 2  <b>  <u> {country_vs_state_option} </u></b></span>",
                    unsafe_allow_html=True)

# Making list of options text (Countries/States)

# getting the countries and states data loaded in option_states from the main_df's state and country columns and
# then adding them in option_states and option_countries in the upper case i.e, us will be added as US...
option_states = [str(x).upper() for x in list(set(main_df['state'])) if str(x) != 'nan']
option_countries = [str(x).upper() for x in list(set(main_df['country'])) if str(x) != 'nan']

# country_vs_state_option is the option selected by the user with the radio button. If countries is choosen then in order
# to display countries, we store option_menu_src as option_countries else it will be option_states
if country_vs_state_option == 'Countries':
    option_menu_src = option_countries
else:
    option_menu_src = option_states

# Options Checkboxes
option_1 = st.sidebar.checkbox(option_menu_src[0], key='option_1', value=True)
option_2 = st.sidebar.checkbox(option_menu_src[1], key='option_2', value=True)
option_3 = st.sidebar.checkbox(option_menu_src[2], key='option_3', )
option_4 = st.sidebar.checkbox(option_menu_src[3], key='option_4', )
option_5 = st.sidebar.checkbox(option_menu_src[4], key='option_5', )

# Min. and Max. Year input boxes
min_year_line_chart = st.sidebar.number_input('Enter Min. Year', value=available_years[0], min_value=available_years[0],
                                              max_value=available_years[-1])
max_year_line_chart = st.sidebar.number_input('Enter Max. Year', value=available_years[-1],
                                              min_value=available_years[0], max_value=available_years[-1])


# main panel starts

# Pie Chart
st.markdown("<h3 style='margin-top:5%'></h3>", unsafe_allow_html=True)
# getting the data from the getPieChartData fn
pie_chart_data = getPieChartData(hemispheres=hemisphere_multi_select)
# piechartdata.values() gives a list of values for spring, summer, fall, winter in a list []
pie_chart_values = pie_chart_data.values()
pie_chart_figure = plt.figure(figsize=(5, 2))
pie_chart_labels = pie_chart_data.keys()
plt.pie(pie_chart_values, labels=pie_chart_labels, textprops={'fontsize': 8}, autopct="%1.1f%%")
plt.title("                     UFO Sightings by Season                    ", fontsize=20)
patches, texts = plt.pie(pie_chart_values)
plt.legend(patches, pie_chart_labels, bbox_to_anchor=(.90, 0.5), loc="center right", fontsize=8,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, )
st.pyplot(pie_chart_figure)

# Stacked Bar Chart and Map Section
stacked_bar_chart_data_container = getStackedBarChartData(list(year_slide_range))

# Map
data = stacked_bar_chart_data_container['map_df']

icon_data = {
    "url": "https://upload.wikimedia.org/wikipedia/commons/8/84/Farm-Fresh_ufo.png",
    "width": 32,
    "height": 32,
    "anchorY": 32,

}

# Adding the icon to dataframe 'data' column 'shape'
for i in data.index:
    data["shape"][i] = icon_data

# plotting the pydeck chart
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    # If want to set an initial state of the map un comment the following lines and set the latitude and longitude value
    #  initial_view_state=pdk.ViewState(
    #      latitude=37.76,
    #      longitude=-122.4,
    #      zoom=4,
    #      pitch=50,
    #  ),

    # defining the layer. In pydeck, layer is used to tell the type of map. Here we used IconLayer to set the icons. get_icon = "shape",here shape is
    # the column of our dataframe "data" which we previously set to the icon_data. Rest are the attributes to define the map.
    layers=[
        pdk.Layer(
            type='IconLayer',
            data=data,
            get_icon="shape",
            get_position='[lon, lat]',
            get_size=4,
            size_scale=15,
            pickable=True,
        ),
    ],
))

# Stacked bar chart
stacked_bar_chart_figure = plt.figure(figsize=(6, 3), )
N = 4
barWidth = .5
xloc = range(N)

# """Plotting the stack bar chart. We have 5 countries stored in stacked_bar_chart_data_container['countries'].
#     So for each country, we get the data of shapes in the currect_bar
#     The stacked_bar_chart_data_container['shape_container] contains a dictionary of shapes and there respective
#     occurence in each country which is indicated by the index. Then comes the plotting part. Here we
#     simply plot the bar if it is the first country otherwise if the country is other then the first country then the
#     bar starts from the sum of the previous countries bar i.e, each country's bar is started from the position where the
#     last countries bar has ended in the graph"""

for index in range(len(stacked_bar_chart_data_container['countries'])):
    currect_bar = [list(x.values())[0][index] for x in stacked_bar_chart_data_container['shape_container']]
    if index == 0:
        plt.bar(xloc, np.array(currect_bar), width=barWidth)
    else:
        previous_lists = [[list(x.values())[0][sub_index] for x in stacked_bar_chart_data_container['shape_container']]
                          for sub_index in range(index)]
        previous_lists = np.array(sum([np.array(x) for x in previous_lists]))
        plt.bar(xloc, np.array(currect_bar), bottom=previous_lists, width=barWidth)

plt.ylabel('UFO frequency')
plt.xlabel('UFO Shapes')
plt.title('                          Top 4 Common UFO Shapes                          ', fontsize=20)
plt.xticks(xloc, [list(x.keys())[0] for x in stacked_bar_chart_data_container['shape_container']])
countries = stacked_bar_chart_data_container['countries']
plt.legend(tuple(countries), bbox_to_anchor=(1.02, 0.5), loc="center right", fontsize=7,
           bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, )
st.pyplot(stacked_bar_chart_figure)

line_chart_data = main_df[(main_df.year >= min_year_line_chart) & (main_df.year <= max_year_line_chart)]
select_options_container = [option_1, option_2, option_3, option_4, option_5]

# """Valid option indexes only allows only 2 countries to be shown in the line chart. It stores the index + 1 (+1 is however substracted later)
#  position of the selected options from the country checkboxes."""

valid_option_indexes = [index + 1 for index in range(len(select_options_container)) if select_options_container[index]][
                       :2]

invalid_option_indexes = [index + 1 for index in range(len(select_options_container)) if
                          index + 1 not in valid_option_indexes]

# Restricting selection of only 2 options
if len([x for x in select_options_container if x]) > 2:
    st.warning(f'Cannot select more than 2 {country_vs_state_option}')

# Line chart
line_chart_figure = plt.figure(figsize=(6, 3), )
line_chart_x_axis = range(int(year_slide_range[0]), int(year_slide_range[1]) + 1)

# """Now for the line chart. As we know line chart will contain data for only 2 countries or 2 states. So first we have to
# see  whether the user has selected countries or state for this chart. If the country is selected, then we will use the
# selected country records for country column of our dataframe else we will use the selected state records(rows) for state column. Next,
# We iterate through all the years between the range specified and check how many times our selected records contains that year.(shape[0]
# indicates the no of records(rows)). Then we simply plot the graph with plt.plot and necessary arguments."""

for valid_index in valid_option_indexes:
    required_key = option_menu_src[valid_index - 1]
    if country_vs_state_option == 'Countries':
        temp_df_line_chart = main_df[(main_df.country == required_key.lower())]
    else:
        temp_df_line_chart = main_df[(main_df.state == required_key.lower())]

    temp_line_chart_data_container = []
    for year in range(int(year_slide_range[0]), int(year_slide_range[1]) + 1):
        temp_line_chart_data_container.append(temp_df_line_chart[(temp_df_line_chart.year == year)].shape[0])
    plt.plot(line_chart_x_axis, temp_line_chart_data_container, label=required_key, linestyle="-")

plt.ylabel('UFO frequency')
plt.xlabel('Year')
plt.title('                                1V1 Comparison                                ', fontsize=20)

# """Including the legends in the line graph. The option_menu_src is the states or countries choosen by the user for this graph.
# tuple is used here to convert the list [a,b,c] to tuple (a,b,c). option_menu_src[index] is the option choosen by the user.
# bbox_to_anchor creates a bounding box of 1.02 width and 0.5 height at the location center right with fontsize 7."""

plt.legend(
    tuple([option_menu_src[index] for index in range(len(option_menu_src)) if index + 1 in valid_option_indexes]),
    bbox_to_anchor=(1.02, 0.5), loc="center right", fontsize=7, bbox_transform=plt.gcf().transFigure)

plt.subplots_adjust(left=0.0, bottom=0.1, )

st.pyplot(line_chart_figure)
