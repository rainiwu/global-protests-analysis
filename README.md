# Global-Protests-Analysis

ECE143 Group 16 Final Project

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#Summary">Summary</a>
    </li>
    <li><a href="#Modules-Packages">Modules and Packages</a></li>
    <li><a href="#How-to-Run-the-Code">How to Run the Code</a></li>
    <li><a href="#File-Structure">File Structure</a></li>
  </ol>
</details>

We use the Mass Mobilization Protest Data, by David Clark and Patrick Regan. It is collected from 162 countries in the time period (1990 - 2020). It contains information about each protest, including protester demands and state responses. 

Mass Mobilization Protest [Dataset Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HTTWYL)

<!-- Summary -->
### Summary
We are analyzing global protests in order to reach our goal of reducing police brutality and help protesters achieve their needs. Our objective of this project is perform some analysis of protests around the world which may bring new insights like: reasons for protests, causes of violence, success factors, global factors, prediction of success

<!-- Modules/Packages -->
### Modules and Packages
1. numpy
2. pandas 
3. matplotlib
4. seaborn
5. pygal - pygal_maps_world
7. plotly
8. cairo - cairosvg
  
<!-- How to Run the Code -->
### How to Run the Code
1. Install python V3
2. Install required modules/packages specified in the Modules/Packages section
3. Run the main file (main.py)
4. Run the Jupyter Notebooks (visualizations.ipynb)

<!-- File Structure -->
### File Structure 
1. src code
   ```sh
   * main.py - main file contains all analysis of the project
   * visualizations.ipynb - containls all visualizations to generate our graphs 
   ```
2. data
   ```sh
   * Original Data - mmALL_073120_csv.csv
   * Cleaned Data(processed) - main_data.csv
   ```
