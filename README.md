# Background

Professional tennis is generally played on three surfaces: hard court, clay, and grass.
### Clay Courts
![](images/clay.jpg)

Clay courts are made from crushed shale. They are categorized as a slow surface, with higher bounces and high responsiveness to spin.
### Hard Courts
![](images/hard.jpg)

Hard courts are constructed from concrete or asphalt with layers of acrylic paint. They are generally considered a moderate/fast paced surface. The pace of the court can be adjusted based on acrylic paint used.

### Grass Courts
![](images/grass.jpg)

Grass courts are considered to be a fast surface with low, fast bounces.

[*images/info source*](https://www.itftennis.com/technical/facilities/facilities-guide/surface-descriptions.aspx)
# Data

The data used in this the project are from [The Tennis Match Charting Project](https://www.kaggle.com/ryanthomasallen/tennis-match-charting-project). The repository contains .csv files with point-by-point data on ATP and WTP tennis matches. The collection of match data was collected through volunteers charting each point of a match according to the format indicated by the creator of the repository (Jeff Sackman).  

Since the data are created on an open source platform, there are differing levels of competence among the volunteer data collectors (match charters). Not all matches are charted with equal complexity. Some matches have detailed accounts of each point's shot type and direction. However, it is more often the case that the match data contains the basic information on how the point began (first serve in vs. second serve in vs. double fault) and how the point ended (winner vs. unforced error). For this project, I selected features that had the most complete data and chose to exclude features with many missing values.

The dataset I use for analysis contains aggregate point information from 510 ATP matches (men's matches only).

# Project Goal

The goal of this project is to use logistic regression to determine whether or not a match was played on a clay court. In this case, the response variable is the odds that a match was played on clay. The predictor variables are various continuous, normally distributed match analytics which (theoretically) could differentiate a match played on a slow vs. fast surface.

# EDA

## Areas for Improvement

1) Include a feature to indicate the length of a point in number of shots per point.

2) Experiment with different thresholds to improve model.

3) Introduce more data to the dataset by also using WTA matches (include a feature for men's vs. women's matches).
