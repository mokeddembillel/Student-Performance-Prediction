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



# Encoding Categorical Data
labelEncoder_X = LabelEncoder()

variables_to_labelEncode = [0, 1, 3, 4, 5, 8, 9 ,10 , 11, 15, 16, 17, 18, 19, 20, 21, 22]

for i in variables_to_encode:
    X.iloc[:, i] = labelEncoder_X.fit_transform(X.iloc[:, i])

# One Hot Encoding
oneHotEncoder_X = OneHotEncoder(sparse = False)

X['Mjob_at_home'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 0]
X['Mjob_health'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 1]
X['Mjob_other'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 2]
X['Mjob_services'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 3]

X['Fjob_at_home'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 0]
X['Fjob_health'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 1]
X['Fjob_other'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 2]
X['Fjob_services'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 3]

X['reason_course'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 0]
X['reason_other'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 1]
X['reason_home'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 2]

X['guardian_father'] = oneHotEncoder_X.fit_transform(X[['guardian']])[:, 0]
X['guardian_mother'] = oneHotEncoder_X.fit_transform(X[['guardian']])[:, 1]

# Drop old columns
X.drop(['Mjob', 'Fjob', 'reason', 'guardian'],axis='columns', inplace=True)

# Reorder columns
X = X[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
      'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
      'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services',
      'reason_course', 'reason_other', 'reason_home', 'guardian_father',
       'guardian_mother','traveltime', 'studytime', 'failures', 'schoolsup', 
       'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]










































