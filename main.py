import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
from geopy.geocoders import Nominatim

# Initial data loading and preprocessing
df = pd.read_csv("cab_data.csv")
df['Date/Time'] = pd.to_datetime(df['Date/Time'])  # Convert to datetime
df['Day_of_Week'] = df['Date/Time'].dt.dayofweek  # Extract day of week
df['Hour'] = df['Date/Time'].dt.hour  # Extract hour
df_clean = df.drop(['Date/Time', 'Base'], axis=1)  # Remove unnecessary columns

# Sample data preparation for initial visualization
sample = df.sample(10000)  # Take random sample of 10,000 rows
sample = sample.sort_values("Hour", ascending=True)  # Sort by hour

# K-means clustering on full cleaned dataset
sample = df_clean.sample(300000)  # Larger sample for clustering
scaler = StandardScaler()
X_all = scaler.fit_transform(sample)  # Standardize features
kmeans = KMeans(n_clusters=12, random_state=0)  # Initialize K-means with 12 clusters
kmeans.fit(X_all)  # Fit K-means model
cluster_centers_all = scaler.inverse_transform(kmeans.cluster_centers_)  # Get cluster centers
c = kmeans.predict(X_all)  # Predict cluster assignments
df_cleaned_all = sample.copy()
df_cleaned_all['cluster_id'] = c  # Add cluster IDs to dataframe
figure_sample = df_cleaned_all.sample(20000)  # Sample for visualization

# DBSCAN clustering
db = DBSCAN(eps=0.20, min_samples=20)  # Initialize DBSCAN
db.fit(X_all)  # Fit DBSCAN model
sample_db = sample.copy()
sample_db['Cluster_id'] = db.labels_  # Assign cluster labels
sample_db['Cluster_id'].value_counts() / sample_db.shape[0] * 100  # Calculate cluster percentages

# Filter out noise points and prepare data for hourly analysis
sample_db_f = sample_db.loc[sample_db['Cluster_id'] >= 0, ['Lat', 'Lon', 'Day_of_Week', 'Hour']]
cluster_counts = sample_db['Cluster_id'].value_counts()
num_noise_points = (sample_db['Cluster_id'] == -1).sum()
percent_noise_points = num_noise_points / sample_db.shape[0] * 100

# Elbow method for determining optimal number of clusters
wcss = []  # Within-cluster sum of squares
k = []  # Number of clusters
h = 12  # Specific hour to analyze
scaler_2 = StandardScaler()
sample_db_f = sample_db_f.sort_values('Hour')
X_h = sample_db_f.loc[sample_db_f['Hour'] == h, ['Lat', 'Lon']]
X_h_n = scaler_2.fit_transform(X_h)

for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_h)
    wcss.append(kmeans.inertia_)
    k.append(i)

# Silhouette score analysis for cluster validation
sil = []  # Silhouette scores
k = []
for i in range(2, 20):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_h_n)
    sil.append(silhouette_score(X_h_n, kmeans.predict(X_h_n)))
    k.append(i)

# Hourly hotspot analysis
scaler_2 = StandardScaler()
labels_hour = []
coord_hour = {}  # Store cluster coordinates by hour
cluster_size = {}  # Store cluster sizes by hour
sample_db_f = sample_db_f.sort_values('Hour')
percentage_clusters = {}  # Store percentage of points in each cluster

for i in range(0, 24):
    X_i = sample_db_f.loc[sample_db_f['Hour'] == i, ['Lat', 'Lon']]
    X_i_n = scaler_2.fit_transform(X_i)
    kmeans = KMeans(n_clusters=12, random_state=0)
    kmeans.fit(X_i_n)
    labels = kmeans.labels_.tolist()
    X_i['cluster_id'] = labels
    coordinates = kmeans.cluster_centers_
    coord_hour[i] = scaler_2.inverse_transform(coordinates)
    cluster_counts = X_i['cluster_id'].value_counts(sort=False).sort_index()
    cluster_size[i] = cluster_counts.to_numpy()
    total_points = cluster_counts.sum()
    percentage_clusters[i] = (cluster_counts / total_points * 100).to_numpy()
    labels_hour.extend(labels)

# Prepare data for hourly hotspot visualization
clusters = np.array([[0, 0, 0, 0]])
for i in range(0, 24):
    cluster_data = np.concatenate([np.full((12, 1), i), coord_hour[i], cluster_size[i].reshape(12, 1)], axis=1)
    clusters = np.concatenate([clusters, cluster_data], axis=0)

clusters = np.delete(clusters, 0, 0)
hour_hotspots = pd.DataFrame(clusters, columns=['Hour', 'Lat', 'Lon', 'Size'])

# Reverse geocoding for location names
neighborhood = []
geolocator = Nominatim(user_agent="NY_hotspots")
for lat, lon in hour_hotspots.loc[:, ['Lat', 'Lon']].to_numpy():
    location = geolocator.reverse(f"{lat:.3f}, {lon:.3f}")
    try:
        neighborhood.append(location.raw['address']['neighbourhood'])
        continue
    except KeyError:
        pass
    # Additional try-except blocks for other address types...
    neighborhood.append(f"{lat} {lon}")

hour_hotspots['Location'] = neighborhood

# Print cluster percentages by hour
for hour, percentages in percentage_clusters.items():
    print(f"Hour {hour}:")
    for cluster_id, percentage in enumerate(percentages):
        print(f"  {hour_hotspots.iloc[cluster_id]['Location']}: {percentage:.2f}%")

# Create animated hourly hotspot map
color_sequence = px.colors.qualitative.Plotly
fig = px.scatter_mapbox(hour_hotspots, lat='Lat', lon='Lon', size='Size', color='Size', 
                       hover_name='Location', animation_frame='Hour',
                       size_max=50, mapbox_style="carto-darkmatter", zoom=10,
                       width=1000, height=700)
fig.update_layout(margin={"r": 0, "t": 150, "l": 0, "b": 0})
fig.write_html("Hourly.html")
fig.show()

# Weekly hotspot analysis
scaler_3 = StandardScaler()
labels_week = []
coord_week = {}  # Store cluster coordinates by day
cluster_size_week = {}  # Store cluster sizes by day
sample_db_f = sample_db_f.sort_values('Day_of_Week')

for i in range(0, 7):
    X_i = sample_db_f.loc[sample_db_f['Day_of_Week'] == i, ['Lat', 'Lon']]
    X_i_n = scaler_3.fit_transform(X_i)
    kmeans = KMeans(n_clusters=12, random_state=0)
    kmeans.fit(X_i_n)
    labels = kmeans.labels_.tolist()
    X_i['cluster_id'] = labels
    coordinates = kmeans.cluster_centers_
    coord_week[i] = scaler_3.inverse_transform(coordinates)
    cluster_size_week[i] = X_i['cluster_id'].value_counts(sort=False).sort_index().to_numpy()
    labels_week.extend(labels)

# Prepare data for weekly hotspot visualization
clusters_week = np.array([[0, 0, 0, 0]])
for i in range(0, 7):
    cluster_data = np.concatenate([np.full((12, 1), i), coord_week[i], cluster_size_week[i].reshape(12, 1)], axis=1)
    clusters_week = np.concatenate([clusters_week, cluster_data], axis=0)

clusters_week = np.delete(clusters_week, 0, 0)
week_hotspots = pd.DataFrame(clusters_week, columns=['Day_of_Week', 'Lat', 'Lon', 'Size'])

# Reverse geocoding for weekly hotspots
geolocator = Nominatim(user_agent="NY_hotspots")
neighborhood = []
for lat, lon in week_hotspots.loc[:, ['Lat', 'Lon']].to_numpy():
    location = geolocator.reverse(f"{lat:.3f}, {lon:.3f}")
    try:
        neighborhood.append(location.raw['address']['neighbourhood'])
        continue
    except KeyError:
        pass
    # Additional try-except blocks for other address types...
    neighborhood.append(f"{lat:.3f}, {lon:.3f}")

week_hotspots['Location'] = neighborhood

# Print cluster percentages by day of week
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
total_bookings_per_day = sample_db_f['Day_of_Week'].value_counts().sort_index().to_numpy()
for i in range(7):
    print(f"{weekday_names[i]}: ")
    cluster_percentages = (cluster_size_week[i] / total_bookings_per_day[i]) * 100
    for cluster_id, percentage in enumerate(cluster_percentages):
        location_name = week_hotspots[
            (week_hotspots['Day_of_Week'] == i) & (week_hotspots['Size'] == cluster_size_week[i][cluster_id])][
            'Location'].values[0]
        print(f"  {location_name}: {percentage:.2f}%")
    print()

# Create animated weekly hotspot map
final = px.scatter_mapbox(week_hotspots, lat='Lat', lon='Lon', size='Size', color='Size',
                         hover_name='Location', animation_frame='Day_of_Week', size_max=50,
                         mapbox_style="carto-darkmatter", zoom=10,
                         width=1000, height=700)
final.update_layout(margin={"r": 0, "t": 150, "l": 0, "b": 0})
final.write_html("Weekly.html")
final.show()
