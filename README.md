This repository contains scripts to find coordination environments of interest.  This repository is under development. The following scripts are available now.
<br/>

1. plot_capped_prisms/ - find capped-prism environments in bi-layered compounds. <br><br> Install the required libraries listed in requirements.txt and then type ```python overlay_plot_capped_prisms.py path prism_nums``` to get started. <br> Type ```python overlay_plot_capped_prisms.py -h``` to get an overview of inputs and options.  <br>The path can be either a single .cif file or a directory containing multiple cif files.
sgs.csv contains the Hermann–Mauguin notation and their latex formatting. Extend this list as required.
colors.csv contains the assigned color-group for each element. Extend this list as required.
<br>
2. inner_CN/ finds the inner coordination numbers using the d/d_min method. This code can run in parallel and the output containing filename, center atom, its Wyckoff, neghbor formula will be saved as .csv file.
<br>
<br>
3. linear_groups - finds the linear M-M-M units made of same element.
