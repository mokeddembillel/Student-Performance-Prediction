# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing The Dataset
dataset = pd.read_csv('./Data/student-mat.csv', sep=';')
X = dataset.iloc[:, 0:32]
X2 = dataset.iloc[:, 0:32]
y = dataset.iloc[:, 32]

# Even if it's not really necessary i did an analysis of this data in PowerBi
# here Are some information to Consider
# -Grades in average are better :
#    -for students in the schoole of "GP" - Gabriel Pereira, 
#    -for students living in "U" - urban, 
#    -for family size "LE3" - less or equal to 3, 
#    -for parent's cohabitation status of "A" - apart, 
#    -for students who has fathers  work as teacher and mothers work in health, 
#    -for students whom choose their schools because of reputation and other reasons
#    -for students whom their gardian is their father 
#    -for students whom their home to school travel time is 1 - <15 min.
#    -for students whom study the most 
#    -for students whom have no extra educational support
#    -for students whom have no family educational support
#    -for students whom have extra paid classes within the course subject
#    -for students whom do extra-curricular activities 
#    -for students whom attended nursery school
#    -for students whom wants higher education
#    -for students whom have internet access
#    -for students with a romantic relationship
#    -for students whom have a free time after school of (from  1 to 5, 2 and 5 are the best)
#    -for students whom go out with friends a little bit (from  1 to 5, 2 is the best)
#    -for students with very low consumption rate (from  1 to 5, 1 is the best)
#    -for students whom are the most healthy
#    ### Most of these variables we stated above has a similar impact on the grades 
#    ### so we will try to do some dimentionality reduction
# -Best Grades in average are of males and students who are 20 years old 
# -males are 187 (47.34%) and females are 208 (52.66%)
# -the quality of family relationships doesn't have a straight impact or a big impact on grades, i will further check later
# -the number of absenses doesn't show w straight impact on the grades 










































