import random

class KMeansClusterClassifier:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centroids = None
        self.clusters = None
        self.labels = None
    
    def random_centroids(self, data):
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(random.choice(data[:-1]))
            
        return centroids
    
    def calculate_distance(self, point1, point2):
        return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5
    
    def assign_clusters(self, data, centroids):
        clusters = [[] for _ in range(len(centroids))]

        for point in data:
            min_distance = float('inf')
            closest_cluster = -1

            for i, centroid in enumerate(centroids):
                distance = self.calculate_distance(point[:-1], centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i

            clusters[closest_cluster].append(point)

        return clusters 
    
    def calculate_new_centroid(self, cluster):
        if not cluster:
            return []
                
        num_features = len(cluster[0]) - 1
        new_centroid = [0] * num_features
        
        for point in cluster:
            for i, feature in enumerate(point[:-1]):
                new_centroid[i] += feature
                
        new_centroid = [feature / len(cluster) for feature in new_centroid]
        return new_centroid
    
    def update_centroids(self, clusters):
        new_centroids = []
        for cluster in clusters:
            new_centroid = self.calculate_new_centroid(cluster)
            new_centroids.append(new_centroid)
        
        return new_centroids
        
    def fit(self, x, y, max_iterations=1000):
        data = x
        
        new_data = []
        for item in data:
            new_data.append(list(item))
            
        for x, y in zip(new_data, y):
            x.append(y)
        
        self.centroids = self.random_centroids(new_data)
        self.clusters = self.assign_clusters(new_data, self.centroids)
        for i in range(max_iterations):
            ex_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters)
            self.clusters = self.assign_clusters(new_data, self.centroids)
            if ex_centroids == self.centroids:
                break
            
        self.labels = self.find_cluster_class_labels()
            
    def find_most_common_class(self, cluster):
        class_counts = {}
        
        for point in cluster:
            class_label = point[-1]  
            class_counts[class_label] = class_counts.get(class_label, 0) + 1
        
        most_common_class = max(class_counts, key=class_counts.get)
        return most_common_class
    
    def find_cluster_class_labels(self):
        class_labels = []

        for cluster in self.clusters:
            if cluster:
                most_common_class = self.find_most_common_class(cluster)
                class_labels.append(most_common_class)

        return class_labels
        
    def predict(self, x):
        data = x
        predictions = []

        for point in data:
            min_distance = float('inf')
            closest_cluster = -1

            for i, centroid in enumerate(self.centroids):
                distance = self.calculate_distance(point[:-1], centroid)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = i

            predicted_class = self.labels[closest_cluster]
            predictions.append(predicted_class)

        return predictions
    
    def calculate_inertia(self):
        inertia = 0
        for cluster_idx, cluster in enumerate(self.clusters):
            centroid = self.centroids[cluster_idx]
            for point in cluster:
                distance_squared = self.calculate_distance(point[:-1], centroid)
                inertia += distance_squared
        return inertia