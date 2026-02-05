import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Model1Trainer:
    """
    Complete training pipeline for Model-1: User Behavior Profiling
    Uses ensemble of KMeans + GMM for robust clustering
    """
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans_model = None
        self.gmm_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.feature_importance = {}
        self.cluster_profiles = {}
        self.training_metrics = {}
        
    def load_features(self, features_path='model1_features.csv'):
        """Load engineered features"""
        print("Loading engineered features...")
        features_df = pd.read_csv(features_path)
        print(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
        return features_df
    
    def preprocess_features(self, features_df):
        """Preprocess features for clustering"""
        print("\nPreprocessing features...")
        
        # Handle missing values
        features_df = features_df.fillna(features_df.median())
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        # Check for any remaining issues
        print(f"Missing values after cleaning: {features_df.isnull().sum().sum()}")
        print(f"Infinite values after cleaning: {np.isinf(features_df).sum().sum()}")
        
        return features_df
    
    def find_optimal_clusters(self, scaled_features, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\nFinding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_scores = []
        
        k_range = range(2, max_clusters + 1)
        
        for k in k_range:
            # KMeans clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, labels))
            davies_bouldin_scores.append(davies_bouldin_score(scaled_features, labels))
            calinski_scores.append(calinski_harabasz_score(scaled_features, labels))
            
            print(f"k={k}: Silhouette={silhouette_scores[-1]:.3f}, "
                  f"Davies-Bouldin={davies_bouldin_scores[-1]:.3f}, "
                  f"Calinski-Harabasz={calinski_scores[-1]:.1f}")
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Elbow curve
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        # Silhouette score (higher is better)
        axes[0, 1].plot(k_range, silhouette_scores, 'go-')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Analysis (Higher is Better)')
        axes[0, 1].grid(True)
        
        # Davies-Bouldin score (lower is better)
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'ro-')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Davies-Bouldin Score')
        axes[1, 0].set_title('Davies-Bouldin Index (Lower is Better)')
        axes[1, 0].grid(True)
        
        # Calinski-Harabasz score (higher is better)
        axes[1, 1].plot(k_range, calinski_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Calinski-Harabasz Score')
        axes[1, 1].set_title('Calinski-Harabasz Index (Higher is Better)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('cluster_optimization_metrics.png', dpi=300)
        print("\nCluster optimization plot saved as 'cluster_optimization_metrics.png'")
        
        # Find optimal k (best silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nRecommended number of clusters: {optimal_k}")
        print(f"Best Silhouette Score: {max(silhouette_scores):.3f}")
        
        return optimal_k, {
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'calinski_scores': calinski_scores
        }
    
    def train_ensemble_clustering(self, scaled_features):
        """Train both KMeans and GMM for ensemble clustering"""
        print(f"\nTraining ensemble clustering with {self.n_clusters} clusters...")
        
        # Train KMeans
        print("Training KMeans...")
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=20,
            max_iter=500
        )
        kmeans_labels = self.kmeans_model.fit_predict(scaled_features)
        
        # Train Gaussian Mixture Model
        print("Training Gaussian Mixture Model...")
        self.gmm_model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='full',
            random_state=42,
            n_init=10,
            max_iter=200
        )
        gmm_labels = self.gmm_model.fit_predict(scaled_features)
        
        # Ensemble approach: use agreement or voting
        ensemble_labels = self.create_ensemble_labels(kmeans_labels, gmm_labels)
        
        # Calculate metrics
        metrics = self.calculate_clustering_metrics(scaled_features, ensemble_labels)
        self.training_metrics = metrics
        
        print("\n=== CLUSTERING METRICS ===")
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f} (>0.3 is good)")
        print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f} (lower is better)")
        print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.1f} (higher is better)")
        
        return ensemble_labels, metrics
    
    def create_ensemble_labels(self, kmeans_labels, gmm_labels):
        """Create ensemble labels from KMeans and GMM"""
        # Simple approach: use KMeans as primary, GMM for validation
        # In production, you might want a more sophisticated voting mechanism
        ensemble_labels = kmeans_labels.copy()
        
        # Calculate agreement rate
        agreement_rate = (kmeans_labels == gmm_labels).mean()
        print(f"KMeans-GMM agreement rate: {agreement_rate:.2%}")
        
        return ensemble_labels
    
    def calculate_clustering_metrics(self, scaled_features, labels):
        """Calculate clustering quality metrics"""
        return {
            'silhouette_score': silhouette_score(scaled_features, labels),
            'davies_bouldin_score': davies_bouldin_score(scaled_features, labels),
            'calinski_harabasz_score': calinski_harabasz_score(scaled_features, labels)
        }
    
    def calculate_feature_importance(self, features_df, labels):
        """Calculate feature importance for clustering"""
        print("\nCalculating feature importance...")
        
        importance_scores = {}
        
        for feature in features_df.columns:
            overall_mean = features_df[feature].mean()
            between_cluster_var = 0
            within_cluster_var = 0
            
            for cluster_id in np.unique(labels):
                cluster_data = features_df[labels == cluster_id][feature]
                cluster_mean = cluster_data.mean()
                cluster_size = len(cluster_data)
                
                between_cluster_var += cluster_size * (cluster_mean - overall_mean) ** 2
                within_cluster_var += ((cluster_data - cluster_mean) ** 2).sum()
            
            if within_cluster_var > 0:
                importance = between_cluster_var / within_cluster_var
            else:
                importance = 0
                
            importance_scores[feature] = importance
        
        # Sort by importance
        self.feature_importance = dict(sorted(
            importance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        # Print top 10
        print("\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(list(self.feature_importance.items())[:10]):
            print(f"{i+1}. {feature}: {score:.3f}")
        
        return self.feature_importance
    
    def create_cluster_profiles(self, features_df, labels, original_df=None):
        """Create descriptive profiles for each cluster"""
        print("\nCreating cluster profiles...")
        
        profiles = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_size = cluster_mask.sum()
            
            # Get cluster statistics
            cluster_features = features_df[cluster_mask]
            
            profile = {
                'cluster_id': int(cluster_id),
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(labels) * 100),
                'feature_means': cluster_features.mean().to_dict(),
                'feature_stds': cluster_features.std().to_dict()
            }
            
            # Add original data statistics if available
            if original_df is not None:
                cluster_original = original_df[cluster_mask]
                profile['original_stats'] = {
                    'avg_age': float(cluster_original['age'].mean()) if 'age' in cluster_original else None,
                    'avg_income': float(cluster_original['monthly_income'].mean()) if 'monthly_income' in cluster_original else None,
                    'avg_expenses': float(cluster_original['monthly_expenses'].mean()) if 'monthly_expenses' in cluster_original else None,
                    'avg_risk_tolerance': float(cluster_original['risk_tolerance'].mean()) if 'risk_tolerance' in cluster_original else None
                }
            
            profiles[cluster_id] = profile
        
        self.cluster_profiles = profiles
        
        # Print summary
        print("\n=== CLUSTER DISTRIBUTION ===")
        for cluster_id, profile in profiles.items():
            print(f"Cluster {cluster_id}: {profile['size']} users ({profile['percentage']:.1f}%)")
        
        return profiles
    
    def assign_cluster_names(self, profiles, features_df, labels):
        """Assign meaningful names to clusters based on their characteristics"""
        print("\nAssigning cluster names...")
        
        cluster_names = {}
        
        for cluster_id, profile in profiles.items():
            means = profile['feature_means']
            
            # Determine cluster characteristics
            risk_score = means.get('risk_tolerance_score', 0.5)
            savings_rate = means.get('savings_rate', 0.1)
            investment_activity = means.get('investment_activity', 0.05)
            planning_orientation = means.get('planning_orientation', 2) / 4
            
            # Name assignment logic (based on architecture doc)
            if risk_score < 0.3 and savings_rate > 0.2:
                name = "Conservative Saver"
                description = "Low risk tolerance, high savings rate, prefers safety"
                
            elif risk_score > 0.6 and investment_activity > 0.1:
                name = "Aggressive Investor"
                description = "High risk tolerance, active investor, growth-focused"
                
            elif 0.3 <= risk_score <= 0.6 and planning_orientation > 0.6:
                name = "Balanced Planner"
                description = "Moderate risk, good planning habits, balanced approach"
                
            elif savings_rate < 0.1 and means.get('discretionary_ratio', 0) > 0.3:
                name = "Impulsive Spender"
                description = "Low savings, high discretionary spending, needs guidance"
                
            else:
                name = "Goal-Oriented Optimizer"
                description = "Focused on specific goals, calculated approach"
            
            cluster_names[cluster_id] = {
                'name': name,
                'description': description
            }
            
            print(f"Cluster {cluster_id}: {name} - {description}")
        
        return cluster_names
    
    def visualize_clusters(self, scaled_features, labels, feature_names):
        """Visualize clusters using PCA"""
        print("\nCreating cluster visualizations...")
        
        # Apply PCA for 2D visualization
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(scaled_features)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            features_2d[:, 0], 
            features_2d[:, 1], 
            c=labels, 
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'First Principal Component ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('User Behavior Clusters (PCA Visualization)')
        plt.grid(True, alpha=0.3)
        
        # Add cluster centers
        if self.kmeans_model:
            centers_2d = pca_2d.transform(self.kmeans_model.cluster_centers_)
            plt.scatter(
                centers_2d[:, 0], 
                centers_2d[:, 1], 
                c='red', 
                marker='X', 
                s=200, 
                edgecolors='black',
                label='Cluster Centers'
            )
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('cluster_visualization.png', dpi=300)
        print("Cluster visualization saved as 'cluster_visualization.png'")
        
        # Feature importance heatmap
        self.plot_feature_importance_heatmap(scaled_features, labels, feature_names)
    
    def plot_feature_importance_heatmap(self, scaled_features, labels, feature_names):
        """Plot heatmap of feature importance across clusters"""
        features_df = pd.DataFrame(scaled_features, columns=feature_names)
        
        # Calculate mean feature values per cluster
        cluster_means = []
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_means.append(features_df[cluster_mask].mean())
        
        cluster_means_df = pd.DataFrame(cluster_means, 
                                       index=[f'Cluster {i}' for i in range(len(cluster_means))])
        
        # Plot top 15 most important features
        top_features = list(self.feature_importance.keys())[:15]
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            cluster_means_df[top_features].T, 
            cmap='RdYlGn', 
            center=0,
            annot=True, 
            fmt='.2f',
            cbar_kws={'label': 'Standardized Feature Value'}
        )
        plt.title('Cluster Profiles: Top 15 Features (Standardized)')
        plt.xlabel('Clusters')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('cluster_feature_heatmap.png', dpi=300)
        print("Feature heatmap saved as 'cluster_feature_heatmap.png'")
    
    def save_model(self, output_dir='model1_output'):
        """Save trained model and artifacts"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving model artifacts to '{output_dir}'...")
        
        # Save models
        joblib.dump(self.kmeans_model, f'{output_dir}/kmeans_model.pkl')
        joblib.dump(self.gmm_model, f'{output_dir}/gmm_model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Save metadata
        metadata = {
            'model_version': '1.0',
            'training_date': datetime.now().isoformat(),
            'n_clusters': int(self.n_clusters),
            'training_metrics': convert_to_json_serializable(self.training_metrics),
            'feature_importance': convert_to_json_serializable(self.feature_importance),
            'cluster_profiles': convert_to_json_serializable(self.cluster_profiles)
        }
        
        with open(f'{output_dir}/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model saved successfully to '{output_dir}/'")
        print("Files created:")
        print("  - kmeans_model.pkl")
        print("  - gmm_model.pkl")
        print("  - scaler.pkl")
        print("  - model_metadata.json")
    
    def train_complete_pipeline(self, features_path='model1_features.csv', 
                               original_data_path='synthetic_financial_users.csv',
                               auto_find_clusters=False):
        """Complete training pipeline"""
        print("="*60)
        print("MODEL-1 TRAINING PIPELINE: USER BEHAVIOR PROFILING")
        print("="*60)
        
        # Load features
        features_df = self.load_features(features_path)
        
        # Load original data for context
        try:
            original_df = pd.read_csv(original_data_path)
        except:
            original_df = None
            print("Warning: Could not load original data for context")
        
        # Preprocess
        features_df = self.preprocess_features(features_df)
        
        # Scale features
        print("\nScaling features...")
        scaled_features = self.scaler.fit_transform(features_df)
        
        # Find optimal clusters (optional)
        if auto_find_clusters:
            optimal_k, optimization_results = self.find_optimal_clusters(scaled_features)
            self.n_clusters = optimal_k
        
        # Train ensemble clustering
        labels, metrics = self.train_ensemble_clustering(scaled_features)
        
        # Calculate feature importance
        self.calculate_feature_importance(features_df, labels)
        
        # Create cluster profiles
        self.create_cluster_profiles(features_df, labels, original_df)
        
        # Assign cluster names
        cluster_names = self.assign_cluster_names(self.cluster_profiles, features_df, labels)
        
        # Visualize
        self.visualize_clusters(scaled_features, labels, features_df.columns)
        
        # Save everything
        self.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"\nModel Performance:")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.1f}")
        
        print(f"\nClusters Created: {self.n_clusters}")
        for cluster_id, names in cluster_names.items():
            print(f"  {cluster_id}: {names['name']}")
        
        return {
            'labels': labels,
            'metrics': metrics,
            'cluster_names': cluster_names,
            'feature_importance': self.feature_importance
        }

# Main execution
if __name__ == "__main__":
    # Initialize trainer
    trainer = Model1Trainer(n_clusters=5)
    
    # Run complete training pipeline
    results = trainer.train_complete_pipeline(
        features_path='model1_features.csv',
        original_data_path='synthetic_financial_users.csv',
        auto_find_clusters=False  # Set to True to automatically find optimal k
    )
    
    print("\nTraining results saved. You can now use this model for predictions!")