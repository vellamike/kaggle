import mat_loading as ml

"""
Load data and plot it:
"""
d = ml.load_data()
print d.start
print d.end
fig = ml.smart_plot(d,d.start,3600)
fig.show()
raw_input("$")
