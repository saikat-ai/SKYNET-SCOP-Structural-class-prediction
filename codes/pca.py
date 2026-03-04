import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Step 1: Normalize
scaled_embeddings = StandardScaler().fit_transform(X)
# Step 2: PCA pre-reduction
pca = PCA(n_components=200, random_state=42)
pca_result = pca.fit_transform(scaled_embeddings)
# Step 3: Construct DataFrame
pca_df = pd.DataFrame()
pca_df['PCA-1'] = pca_result[:, 0]
pca_df['PCA-2'] = pca_result[:, 1]
pca_df['Class'] = df['Class'].values
# Step 4: Plot
plt.figure(figsize=(8, 6))
sns.set(style='whitegrid', font_scale=1.2)
# Choose an appealing palette (change 'husl' or try 'Spectral', 'Set2', etc.)
palette = sns.color_palette("husl", len(pca_df['Class'].unique()))
scatter = sns.scatterplot(x='PCA-1', y='PCA-2', hue='Class', palette=palette,data=pca_df, alpha=0.90, s=60, edgecolor='k', linewidth=0.4)
# Style tweaks
plt.title('PCA Projection of K-mer Embedding', fontsize=16, weight='bold', pad=20)
plt.xlabel('Principal Component 1', fontsize=14,fontweight='bold')
plt.ylabel('Principal Component 2', fontsize=14,fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.4)
# Legend outside
plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()
plt.savefig('scoop_kmer_pca.png')
plt.show()
