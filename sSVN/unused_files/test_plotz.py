from plotz import *
from math import sin, pi

with Plot("myfigure") as p:
    p.title = r"My first \texttt{PlotZ} plot"
    p.x.label = "$x$"
    p.y.label = "$y$"
    # p.y.label_rotate = True
    p.style.dashed()
    p.style.colormap("monochrome")
    # p.y.label_rotate = True
    p.plot(Function(sin, samples=50, range=(0, pi)),
           title=r"$\sin(x)$")

    p.legend("north east")