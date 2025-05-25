from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        df = pd.read_csv('HybridDataset.csv')
        
        time_mapping = {
            'Less than 2': 1,
            '2-4': 3,
            '4-6': 5,
            '6-8': 7,
            'More than 8': 9
        }
        df['Time_Spent_Numeric'] = df['how many hours per day do you spend online?'].map(time_mapping)
        
        df['Platform_Count'] = df['what is the total number of social media platforms that you use?'].str.extract('(\d+)').astype(float)
        df.loc[df['what is the total number of social media platforms that you use?'] == '4 +', 'Platform_Count'] = 4
        
        df['Engagement_Score'] = df['which activities do you engage in most on social media?'].str.count('Liking|Sharing|Commenting|Posting|Watching')
        
        df['Engagement_Score'] = df['Engagement_Score'].fillna(0)
        
        features = ['Time_Spent_Numeric', 'Platform_Count', 'Engagement_Score']
        X = df[features].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate inertia for different numbers of clusters
        inertias = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Plot elbow method
        plt.figure(figsize=(10,6))
        plt.plot(K, inertias, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method For Optimal k')
        plt.savefig('static/elbow_plot.png')
        plt.close()

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x='Time_Spent_Numeric', y='Engagement_Score', hue='Cluster', palette='Set1')
        plt.title('Segmentasi Pengguna Media Sosial')
        plt.xlabel('Waktu Penggunaan (jam)')
        plt.ylabel('Skor Keterlibatan')
        
        if not os.path.exists('static'):
            os.makedirs('static')
            
        plt.savefig('static/cluster_plot.png')
        plt.close()

        display_columns = {
            'what is your age group?': 'Age Group',
            'what is your occupation?': 'Occupation',
            'how many hours per day do you spend online?': 'Daily Online Hours',
            'what is the total number of social media platforms that you use?': 'Number of Platforms',
            'Cluster': 'User Segment'
        }
        
        display_df = df[display_columns.keys()].rename(columns=display_columns)
        
        table_html = display_df.to_html(
            classes='table table-striped table-hover table-bordered',
            index=False,
            table_id='results-table',
            justify='center'
        )
        
        return render_template('result.html', 
                             tables=[table_html], 
                             image='static/cluster_plot.png',
                             elbow_plot='static/elbow_plot.png')
                             
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
