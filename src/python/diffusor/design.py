# fmax calculation matches multiple online calculators. Not sure about fmin

def diffusor_bandwidth(well_width, max_depth, c=343.0):
    fmin = c/(max_depth*4)
    fmax = c/(well_width*2)
    return fmin, fmax


def diffusor_dimensions(fmin, fmax, c=343.0):
    max_depth = c/(fmin*4)
    well_width = c/(fmax*2)
    return max_depth, well_width
