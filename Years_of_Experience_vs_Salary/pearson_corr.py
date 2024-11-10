# Helper function to find pearson's correlation manually
def pearson_correalation(x, y):
    n = len(x)
    xy = [x*y for x, y in zip(x, y)]
    x_squared = [x**2 for x in x]
    y_squarred = [y**2 for y in y]
    
    # Calculate Numerator & Denominator 
    numerator = (n * sum(xy)) - (sum(x) * sum(y))
    # Split Denominator into x & y portions then multiply and take square
    x_denom = (n * sum(x_squared) - sum(x)**2)
    y_denom = (n * sum(y_squarred) - sum(y)**2)
    pre_denominatot = x_denom * y_denom
    denominatot = pre_denominatot ** 0.5

    # Return correlation coefficient
    return numerator / denominatot

    