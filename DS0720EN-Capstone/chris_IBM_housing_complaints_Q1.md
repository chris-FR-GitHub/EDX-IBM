 # IBM DS0720EN - Capstone Project
 Chris. -- https://github.com/chris-FR-GitHub/EDX-IBM

## Problem Statement
The people of New Yorker use the 311 system to report complaints about the non-emergency problems to local authorities. Various agencies in New York are assigned these problems. The Department of Housing Preservation and Development of New York City is the agency that processes 311 complaints that are related to housing and buildings.  

In the last few years, the number of 311 complaints coming to the Department of Housing Preservation and Development has increased significantly. Although these complaints are not necessarily urgent, the large volume of complaints and the sudden increase is impacting the overall efficiency of operations of the agency.  

Therefore, the Department of Housing Preservation and Development has approached your organization to help them manage the large volume of 311 complaints they are receiving every year.  

## <font color=blue>Q1 : Which type of complaint should the Department of Housing Preservation and Development of New York City focus on first?</font>

### Preparatory stage

As specified in the **"Ingest the NYC 311 Dataset"** page, we uploaded a **10 million max** records file, containing the following columns : 
*created_date, unique_key, complaint_type, incident_zip, incident_address, street_name, address_type, city, resolution_description, borough, latitude, longitude, closed_date, location_type, status*

We stored this file in the input folder.

We named it : **NYC_311_Dataset.csv**

The file size is around 2.5 Gb.

### Load the data file


```python
# Import required libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
# Load the file in a dataframe
# As suggested in the "Ingest the NYC 311 Dataset" page : Parse the date fields by using the parse_dates option
df_all = pd.read_csv('./input/NYC_311_Dataset.csv', parse_dates = ['created_date', 'closed_date'])
# Display the 10 first rows
df_all.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_date</th>
      <th>unique_key</th>
      <th>complaint_type</th>
      <th>incident_zip</th>
      <th>incident_address</th>
      <th>street_name</th>
      <th>address_type</th>
      <th>city</th>
      <th>resolution_description</th>
      <th>borough</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>closed_date</th>
      <th>location_type</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2020-01-28 20:26:11</td>
      <td>45492325</td>
      <td>HEAT/HOT WATER</td>
      <td>10462.0</td>
      <td>2040 BRONXDALE AVENUE</td>
      <td>BRONXDALE AVENUE</td>
      <td>ADDRESS</td>
      <td>BRONX</td>
      <td>The complaint you filed is a duplicate of a co...</td>
      <td>BRONX</td>
      <td>40.850795</td>
      <td>-73.866537</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2020-01-28 15:57:56</td>
      <td>45493601</td>
      <td>UNSANITARY CONDITION</td>
      <td>11368.0</td>
      <td>55-02 98 STREET</td>
      <td>98 STREET</td>
      <td>ADDRESS</td>
      <td>Corona</td>
      <td>The following complaint conditions are still o...</td>
      <td>QUEENS</td>
      <td>40.738846</td>
      <td>-73.862785</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2020-01-28 04:49:03</td>
      <td>45494360</td>
      <td>HEAT/HOT WATER</td>
      <td>11225.0</td>
      <td>181 HAWTHORNE STREET</td>
      <td>HAWTHORNE STREET</td>
      <td>ADDRESS</td>
      <td>BROOKLYN</td>
      <td>The Department of Housing Preservation and Dev...</td>
      <td>BROOKLYN</td>
      <td>40.657592</td>
      <td>-73.954469</td>
      <td>2020-01-28 21:37:27</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Closed</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2020-01-28 18:07:16</td>
      <td>45493438</td>
      <td>PLUMBING</td>
      <td>10454.0</td>
      <td>514 EAST  138 STREET</td>
      <td>EAST  138 STREET</td>
      <td>ADDRESS</td>
      <td>BRONX</td>
      <td>The following complaint conditions are still o...</td>
      <td>BRONX</td>
      <td>40.807416</td>
      <td>-73.918919</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2020-01-28 14:10:04</td>
      <td>45492347</td>
      <td>HEAT/HOT WATER</td>
      <td>10463.0</td>
      <td>2800 BAILEY AVENUE</td>
      <td>BAILEY AVENUE</td>
      <td>ADDRESS</td>
      <td>BRONX</td>
      <td>The complaint you filed is a duplicate of a co...</td>
      <td>BRONX</td>
      <td>40.873356</td>
      <td>-73.905554</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2020-01-28 08:07:55</td>
      <td>45492310</td>
      <td>HEAT/HOT WATER</td>
      <td>11225.0</td>
      <td>1789 BEDFORD AVENUE</td>
      <td>BEDFORD AVENUE</td>
      <td>ADDRESS</td>
      <td>BROOKLYN</td>
      <td>The following complaint conditions are still o...</td>
      <td>BROOKLYN</td>
      <td>40.662278</td>
      <td>-73.957064</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2020-01-28 21:11:02</td>
      <td>45492265</td>
      <td>HEAT/HOT WATER</td>
      <td>11226.0</td>
      <td>2815 BEVERLY ROAD</td>
      <td>BEVERLY ROAD</td>
      <td>ADDRESS</td>
      <td>BROOKLYN</td>
      <td>The following complaint conditions are still o...</td>
      <td>BROOKLYN</td>
      <td>40.645038</td>
      <td>-73.950632</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2020-01-28 07:27:16</td>
      <td>45493379</td>
      <td>HEAT/HOT WATER</td>
      <td>10462.0</td>
      <td>2040 BRONXDALE AVENUE</td>
      <td>BRONXDALE AVENUE</td>
      <td>ADDRESS</td>
      <td>BRONX</td>
      <td>The complaint you filed is a duplicate of a co...</td>
      <td>BRONX</td>
      <td>40.850795</td>
      <td>-73.866537</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2020-01-28 07:21:15</td>
      <td>45494446</td>
      <td>HEAT/HOT WATER</td>
      <td>10462.0</td>
      <td>2040 BRONXDALE AVENUE</td>
      <td>BRONXDALE AVENUE</td>
      <td>ADDRESS</td>
      <td>BRONX</td>
      <td>The complaint you filed is a duplicate of a co...</td>
      <td>BRONX</td>
      <td>40.850795</td>
      <td>-73.866537</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2020-01-28 07:08:25</td>
      <td>45493589</td>
      <td>PLUMBING</td>
      <td>11201.0</td>
      <td>436 ALBEE SQUARE</td>
      <td>ALBEE SQUARE</td>
      <td>ADDRESS</td>
      <td>BROOKLYN</td>
      <td>The following complaint conditions are still o...</td>
      <td>BROOKLYN</td>
      <td>40.690803</td>
      <td>-73.983478</td>
      <td>NaT</td>
      <td>RESIDENTIAL BUILDING</td>
      <td>Open</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the dataframe size
df_all.shape
```




    (6015270, 15)



### Check NaN values


```python
# Count NaN values in each column
df_NaN_count = df_all.isnull().sum(axis = 0)
df_NaN_count = df_NaN_count.to_frame()
df_NaN_count.rename(columns = {list(df_NaN_count)[0]:'Count'}, inplace=True)
df_NaN_count['NaN %'] = df_NaN_count['Count'] / len(df_all)
df_NaN_count['NaN %'] = df_NaN_count['NaN %'].map(lambda x: "{0:.2f}%".format(x*100))
df_NaN_count
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>NaN %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>created_date</td>
      <td>0</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <td>unique_key</td>
      <td>0</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <td>complaint_type</td>
      <td>0</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <td>incident_zip</td>
      <td>80704</td>
      <td>1.34%</td>
    </tr>
    <tr>
      <td>incident_address</td>
      <td>52831</td>
      <td>0.88%</td>
    </tr>
    <tr>
      <td>street_name</td>
      <td>52831</td>
      <td>0.88%</td>
    </tr>
    <tr>
      <td>address_type</td>
      <td>84772</td>
      <td>1.41%</td>
    </tr>
    <tr>
      <td>city</td>
      <td>80281</td>
      <td>1.33%</td>
    </tr>
    <tr>
      <td>resolution_description</td>
      <td>7828</td>
      <td>0.13%</td>
    </tr>
    <tr>
      <td>borough</td>
      <td>0</td>
      <td>0.00%</td>
    </tr>
    <tr>
      <td>latitude</td>
      <td>80678</td>
      <td>1.34%</td>
    </tr>
    <tr>
      <td>longitude</td>
      <td>80678</td>
      <td>1.34%</td>
    </tr>
    <tr>
      <td>closed_date</td>
      <td>126969</td>
      <td>2.11%</td>
    </tr>
    <tr>
      <td>location_type</td>
      <td>52830</td>
      <td>0.88%</td>
    </tr>
    <tr>
      <td>status</td>
      <td>0</td>
      <td>0.00%</td>
    </tr>
  </tbody>
</table>
</div>



No NaN values for the **complaint_type** and **status** columns.  
Some NaN in the **closed_date** column but it makes sense as only closed complaints have a closed date.


```python
print ('Only' , df_all[df_all['status']== 'Closed']['closed_date'].isnull().sum(axis = 0), 
       'of', len(df_all[df_all['status']== 'Closed']), 'closed records have no closed date')
```

    Only 197 of 5881368 closed records have no closed date
    

### Check unique values


```python
#Check distinct complaint_type
df_all['complaint_type'].unique()
```




    array(['HEAT/HOT WATER', 'UNSANITARY CONDITION', 'PLUMBING',
           'PAINT/PLASTER', 'DOOR/WINDOW', 'APPLIANCE', 'WATER LEAK',
           'ELECTRIC', 'GENERAL', 'FLOORING/STAIRS', 'SAFETY', 'ELEVATOR',
           'OUTSIDE BUILDING', 'Unsanitary Condition',
           'HPD Literature Request', 'HEATING', 'PAINT - PLASTER', 'Safety',
           'Electric', 'NONCONST', 'CONSTRUCTION', 'GENERAL CONSTRUCTION',
           'General', 'AGENCY', 'STRUCTURAL', 'VACANT APARTMENT',
           'Outside Building', 'Plumbing', 'Appliance', 'Mold'], dtype=object)




```python
#Check distinct Status
df_all['status'].unique()
```




    array(['Open', 'Closed', 'Assigned', 'In Progress', 'Pending'],
          dtype=object)



### Count by complaint_type


```python
def getCountAndPercent(col):
    df = pd.concat([col.value_counts(),              
                    col.value_counts(normalize=True).mul(100).map(lambda x: "{0:.2f}%".format(x))],
                   axis=1,
                   keys=('Count','Percentage'))
    return df

#get the count and percentage
df_complaint_type = getCountAndPercent( df_all['complaint_type'])
df_complaint_type.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HEAT/HOT WATER</td>
      <td>1258260</td>
      <td>20.92%</td>
    </tr>
    <tr>
      <td>HEATING</td>
      <td>887869</td>
      <td>14.76%</td>
    </tr>
    <tr>
      <td>PLUMBING</td>
      <td>710913</td>
      <td>11.82%</td>
    </tr>
    <tr>
      <td>GENERAL CONSTRUCTION</td>
      <td>500863</td>
      <td>8.33%</td>
    </tr>
    <tr>
      <td>UNSANITARY CONDITION</td>
      <td>451299</td>
      <td>7.50%</td>
    </tr>
    <tr>
      <td>PAINT - PLASTER</td>
      <td>361257</td>
      <td>6.01%</td>
    </tr>
    <tr>
      <td>PAINT/PLASTER</td>
      <td>346329</td>
      <td>5.76%</td>
    </tr>
    <tr>
      <td>ELECTRIC</td>
      <td>307214</td>
      <td>5.11%</td>
    </tr>
    <tr>
      <td>NONCONST</td>
      <td>260890</td>
      <td>4.34%</td>
    </tr>
    <tr>
      <td>DOOR/WINDOW</td>
      <td>205140</td>
      <td>3.41%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot the count
def plotTOP10(df, strTitle):
    # TOP 10 and an OTHER type
    top10_other = df.head(10)
    if len(df) > 10:
        top10_other['Remaining {0} types'.format(len(df) - 10)] = sum(df[10:])

    top10_other.plot(kind='barh', figsize=(10, 6),fontsize=12)
    plt.xlabel('Number of Complaints',fontsize=14)
    plt.title(strTitle, fontsize=14)
    plt.gca().invert_yaxis()
    [plt.text(v, i, '{:,d}'.format(v)) for i, v in enumerate(top10_other)];
    plt.show()
    
plotTOP10(df_complaint_type['Count'], 'TOP 10 - Number of records by Complaint type')


```


![png](images/q1_output_18_0.png)


If we follow the Quizz, we can merge the **HEAT/HOT WATER** & **HEATING** types.


```python
# Merge the 'HEAT/HOT WATER' & 'HEATING' types
df_all['complaint_type'].replace(
    to_replace=['HEAT/HOT WATER', 'HEATING'],
    value='HEATING/HOT WATER',
    inplace=True
)
#get the count and percentage
df_complaint_type = getCountAndPercent( df_all['complaint_type'])

df_complaint_type.head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>HEATING/HOT WATER</td>
      <td>2146129</td>
      <td>35.68%</td>
    </tr>
    <tr>
      <td>PLUMBING</td>
      <td>710913</td>
      <td>11.82%</td>
    </tr>
    <tr>
      <td>GENERAL CONSTRUCTION</td>
      <td>500863</td>
      <td>8.33%</td>
    </tr>
    <tr>
      <td>UNSANITARY CONDITION</td>
      <td>451299</td>
      <td>7.50%</td>
    </tr>
    <tr>
      <td>PAINT - PLASTER</td>
      <td>361257</td>
      <td>6.01%</td>
    </tr>
    <tr>
      <td>PAINT/PLASTER</td>
      <td>346329</td>
      <td>5.76%</td>
    </tr>
    <tr>
      <td>ELECTRIC</td>
      <td>307214</td>
      <td>5.11%</td>
    </tr>
    <tr>
      <td>NONCONST</td>
      <td>260890</td>
      <td>4.34%</td>
    </tr>
    <tr>
      <td>DOOR/WINDOW</td>
      <td>205140</td>
      <td>3.41%</td>
    </tr>
    <tr>
      <td>WATER LEAK</td>
      <td>193521</td>
      <td>3.22%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot
plotTOP10(df_complaint_type['Count'], 'TOP 10 - Number of records by Complaint type')
```


![png](images/q1_output_21_0.png)


####  <font color="blue">The Department of Housing Preservation and Development of New York City should address **both of 'Heat/Hot Water' and 'Heating'** complaint types first.</font>

### Extra check : Closed Complaints resolution time (in Days)


```python
df_closed = df_all[df_all['status'] == 'Closed'][['complaint_type', 'created_date','closed_date']]
df_closed.shape
```




    (5881368, 3)




```python
# Compute delta between open and closed dates.
df_closed[['created_date','closed_date']] = df_closed[['created_date','closed_date']].apply(pd.to_datetime)
df_closed['Delta in Days'] = (df_closed['closed_date'] - df_closed['created_date']).dt.days
# drop records containing NaNs
df_closed.dropna(inplace=True)
df_closed.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>complaint_type</th>
      <th>created_date</th>
      <th>closed_date</th>
      <th>Delta in Days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2</td>
      <td>HEATING/HOT WATER</td>
      <td>2020-01-28 04:49:03</td>
      <td>2020-01-28 21:37:27</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>12</td>
      <td>HEATING/HOT WATER</td>
      <td>2020-01-28 16:57:34</td>
      <td>2020-01-28 21:37:43</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>HEATING/HOT WATER</td>
      <td>2020-01-28 16:49:00</td>
      <td>2020-01-28 17:40:01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>35</td>
      <td>HEATING/HOT WATER</td>
      <td>2020-01-28 13:38:52</td>
      <td>2020-01-28 17:40:01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>41</td>
      <td>HEATING/HOT WATER</td>
      <td>2020-01-28 04:45:39</td>
      <td>2020-01-28 15:58:43</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def Percentile25(g):
    return np.percentile(g, 25)

def Percentile75(g):
    return np.percentile(g, 75)

df_closed_avg = df_closed.pivot_table(index=['complaint_type'],
                    values='Delta in Days',
                    aggfunc=('min',Percentile25,'mean','median',Percentile75,'max','count')).reset_index()
df_closed_avg = df_closed_avg[['complaint_type', 'count', 'min', 'Percentile25',  'mean', 'median','Percentile75', 'max']]
# List top 10 complaint types
df_closed_avg.sort_values(by='count', ascending=False).head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>complaint_type</th>
      <th>count</th>
      <th>min</th>
      <th>Percentile25</th>
      <th>mean</th>
      <th>median</th>
      <th>Percentile75</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>12</td>
      <td>HEATING/HOT WATER</td>
      <td>2130132</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.166615</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1686.0</td>
    </tr>
    <tr>
      <td>19</td>
      <td>PLUMBING</td>
      <td>684785</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.129387</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>1207.0</td>
    </tr>
    <tr>
      <td>10</td>
      <td>GENERAL CONSTRUCTION</td>
      <td>471196</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>14.798211</td>
      <td>10.0</td>
      <td>20.0</td>
      <td>733.0</td>
    </tr>
    <tr>
      <td>24</td>
      <td>UNSANITARY CONDITION</td>
      <td>447614</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>22.059891</td>
      <td>13.0</td>
      <td>23.0</td>
      <td>1881.0</td>
    </tr>
    <tr>
      <td>18</td>
      <td>PAINT/PLASTER</td>
      <td>344627</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>17.799331</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>2835.0</td>
    </tr>
    <tr>
      <td>17</td>
      <td>PAINT - PLASTER</td>
      <td>339589</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>14.430515</td>
      <td>10.0</td>
      <td>19.0</td>
      <td>726.0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>ELECTRIC</td>
      <td>297951</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>16.629372</td>
      <td>10.0</td>
      <td>19.0</td>
      <td>1207.0</td>
    </tr>
    <tr>
      <td>14</td>
      <td>NONCONST</td>
      <td>245582</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>18.203065</td>
      <td>11.0</td>
      <td>21.0</td>
      <td>1207.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>DOOR/WINDOW</td>
      <td>203888</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>18.742265</td>
      <td>12.0</td>
      <td>21.0</td>
      <td>1889.0</td>
    </tr>
    <tr>
      <td>27</td>
      <td>WATER LEAK</td>
      <td>192112</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>19.040653</td>
      <td>12.0</td>
      <td>21.0</td>
      <td>1231.0</td>
    </tr>
  </tbody>
</table>
</div>



There are a lot of outliers (see max column).  
The "HEATING/HOT WATER" complaints are closed around the 3 days.  
The other types are closed in 10 to 14 days.  

### Non Closed complaints

If we check the non-closed complaints ('Open', 'Assigned', 'In Progress', 'Pending'), the result is a little bit different.


```python
df_complaint_type_notclosed = getCountAndPercent(df_all[df_all['status']!= 'Closed']['complaint_type'])
df_complaint_type_notclosed.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GENERAL CONSTRUCTION</td>
      <td>29623</td>
      <td>22.12%</td>
    </tr>
    <tr>
      <td>PLUMBING</td>
      <td>26086</td>
      <td>19.48%</td>
    </tr>
    <tr>
      <td>PAINT - PLASTER</td>
      <td>21647</td>
      <td>16.17%</td>
    </tr>
    <tr>
      <td>HEATING/HOT WATER</td>
      <td>15962</td>
      <td>11.92%</td>
    </tr>
    <tr>
      <td>NONCONST</td>
      <td>15278</td>
      <td>11.41%</td>
    </tr>
    <tr>
      <td>ELECTRIC</td>
      <td>9250</td>
      <td>6.91%</td>
    </tr>
    <tr>
      <td>APPLIANCE</td>
      <td>3793</td>
      <td>2.83%</td>
    </tr>
    <tr>
      <td>UNSANITARY CONDITION</td>
      <td>3685</td>
      <td>2.75%</td>
    </tr>
    <tr>
      <td>PAINT/PLASTER</td>
      <td>1702</td>
      <td>1.27%</td>
    </tr>
    <tr>
      <td>WATER LEAK</td>
      <td>1409</td>
      <td>1.05%</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot
plotTOP10(df_complaint_type_notclosed['Count'], 'TOP 10 - status != Closed - Number of records by Complaint type')
```


![png](images/q1_output_30_0.png)


If we only check "Non Closed" complaints, GENERAL CONSTRUCTION type have the more complaints. 

## <font color=blue>Q1 Conclusion</font>

- <font color="blue">The Department of Housing Preservation and Development of New York City should address **both of 'Heat/Hot Water' and 'Heating'** complaint types first. They represent around 2.1 millions complaints (35%).</font>  
- <font color="blue">The total number of *General Construction* complaints is **above 500,000**.</font>  


```python

```


```python

```
