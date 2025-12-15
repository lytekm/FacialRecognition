import numpy as np
import pandas as pd
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.io import loadmat
from sklearn.decomposition import PCA
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, Sequential


def show_raw_face(facedat, person_idx, img_idx=0):
    person_images = facedat[0, person_idx]

    img = person_images[:, :, img_idx]

    img_to_show = img

    plt.imshow(img_to_show, cmap="gray")
    plt.title(f"Person {person_idx}, image {img_idx}\nShape: {img_to_show.shape}")
    plt.axis("off")
    plt.show()


mat_path = "umist_cropped.mat"

# load in the data
data = loadmat(mat_path)

print(data.keys())

# explore the data structure
for k, v in data.items():
    print(f"Key: {k}, Type: {type(v)}, Shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")

print("Entry type:", type(data["facedat"][0, 0]))
print("Entry contents:", data["facedat"][0, 0])

facedat = data["facedat"]
dirnames = data["dirnames"]

print("image shape:", facedat[0][0].shape)
print("image representation: ", facedat[0][0][0])

# show a test face image
show_raw_face(facedat, person_idx=18)

all_images = []
all_labels = []

TARGET_SHAPE = (92, 112)  # assign a height and width so all images are the same size


num_people = facedat.shape[1]
# create a dataframe with all images resized to TARGET_SHAPE
for person_idx in range(num_people):
    person_images = facedat[0, person_idx]
    H, W, N_images = person_images.shape

    for img_idx in range(N_images):
        img = person_images[:, :, img_idx]

        img_resized = resize(img, TARGET_SHAPE, anti_aliasing=True)

        all_images.append(img_resized.flatten())
        all_labels.append(person_idx)

X = np.array(all_images)
df = pd.DataFrame(X)
df["label"] = all_labels

print(df.shape)
print(df.head())

H, W = 92, 112

# show sample image of each person
unique_people = sorted(df["label"].unique())

plt.figure(figsize=(12, 10))

for idx, person in enumerate(unique_people):

    row = df[df["label"] == person].iloc[0]

    img_flat = row.drop("label").values

    img = img_flat.reshape(H, W)

    plt.subplot(4, 5, idx + 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Person {person}")
    plt.axis("off")

plt.tight_layout()
plt.show()

X = df.drop("label", axis=1).values
y = df["label"].values

def compare_distributions(
    y_train_strat, y_val_strat, y_test_strat,
    y_train_ns, y_val_ns, y_test_ns
):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    sns.countplot(x=y_train_strat, ax=axs[0, 0])
    axs[0, 0].set_title("Stratified – Train")
    axs[0, 0].set_xlabel("Person ID")
    axs[0, 0].set_ylabel("Count")

    sns.countplot(x=y_val_strat, ax=axs[0, 1])
    axs[0, 1].set_title("Stratified – Validation")
    axs[0, 1].set_xlabel("Person ID")
    axs[0, 1].set_ylabel("")

    sns.countplot(x=y_test_strat, ax=axs[0, 2])
    axs[0, 2].set_title("Stratified – Test")
    axs[0, 2].set_xlabel("Person ID")
    axs[0, 2].set_ylabel("")

    sns.countplot(x=y_train_ns, ax=axs[1, 0])
    axs[1, 0].set_title("Non-stratified – Train")
    axs[1, 0].set_xlabel("Person ID")
    axs[1, 0].set_ylabel("Count")

    sns.countplot(x=y_val_ns, ax=axs[1, 1])
    axs[1, 1].set_title("Non-stratified – Validation")
    axs[1, 1].set_xlabel("Person ID")
    axs[1, 1].set_ylabel("")

    sns.countplot(x=y_test_ns, ax=axs[1, 2])
    axs[1, 2].set_title("Non-stratified – Test")
    axs[1, 2].set_xlabel("Person ID")
    axs[1, 2].set_ylabel("")

    plt.tight_layout()
    plt.show()


RANDOM_STATE = 42

# Stratified split

X_train_strat, X_temp_strat, y_train_strat, y_temp_strat = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_val_strat, X_test_strat, y_val_strat, y_test_strat = train_test_split(
    X_temp_strat, y_temp_strat,
    test_size=0.50,
    stratify=y_temp_strat,
    random_state=RANDOM_STATE,
)


# Non-stratified split for comparison
X_train_ns, X_temp_ns, y_train_ns, y_temp_ns = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_STATE,
)

X_val_ns, X_test_ns, y_val_ns, y_test_ns = train_test_split(
    X_temp_ns, y_temp_ns,
    test_size=0.50,
    random_state=RANDOM_STATE,
)

# Compare distributions
compare_distributions(
    y_train_strat, y_val_strat, y_test_strat,
    y_train_ns, y_val_ns, y_test_ns
)


scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train_strat)

# Transform val and test
X_val_scaled = scaler.transform(X_val_strat)
X_test_scaled = scaler.transform(X_test_strat)


# Dimensionality Reduction Comparison

#PCA
pca = PCA()
pca.fit(X_train_scaled)

plt.figure(figsize=(10,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance – PCA")
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()

pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_train_pca_2[:,0], X_train_pca_2[:,1], c=y_train_strat, cmap="tab20", s=5)
plt.title("PCA Projection (2 Components)")
plt.colorbar()
plt.show()

# AutoEncoder
input_dim = X_train_scaled.shape[1]
code_dim = 50  # compression size to test

input_layer = Input(shape=(input_dim,))
encoded = Dense(200, activation='relu')(input_layer)
code = Dense(code_dim, activation='relu')(encoded)
decoded = Dense(200, activation='relu')(code)
output = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output)
encoder = Model(input_layer, code)

autoencoder.compile(optimizer=Adam(1e-3), loss="mse")

history = autoencoder.fit(
    X_train_scaled, X_train_scaled,
    validation_data=(X_val_scaled, X_val_scaled),
    epochs=30,
    batch_size=64,
    verbose=1
)

encoded_train = encoder.predict(X_train_scaled)

plt.figure(figsize=(8,6))
plt.scatter(encoded_train[:,0], encoded_train[:,1], c=y_train_strat, cmap="tab20", s=5)
plt.title("Autoencoder Latent Space (First 2 Dimensions)")
plt.colorbar()
plt.show()

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_train_tsne[:,0], X_train_tsne[:,1], c=y_train_strat, cmap="tab20", s=5)
plt.title("t-SNE Visualization")
plt.colorbar()
plt.show()


umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_train_umap = umap_model.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_train_umap[:,0], X_train_umap[:,1], c=y_train_strat, cmap="tab20", s=5)
plt.title("UMAP Projection")
plt.colorbar()
plt.show()

x_train_encoded = encoder.predict(X_train_scaled)
x_val_encoded = encoder.predict(X_val_scaled)
x_test_encoded = encoder.predict(X_test_scaled)


# clustering on encoded data
kmeans = KMeans(n_clusters=num_people, random_state=42)
kmeans_labels = kmeans.fit_predict(x_train_encoded)

dbscan = DBSCAN(eps=3, min_samples=5)
db_scan_labels = dbscan.fit_predict(x_train_encoded)

agg = AgglomerativeClustering(n_clusters=20, linkage='ward')
agg_labels = agg.fit_predict(x_train_encoded)

# Visualize clustering results with t-SNE
tsne_encoded = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
x_train_tsne_encoded = tsne_encoded.fit_transform(x_train_encoded)
plt.figure(figsize=(8,6))
plt.scatter(x_train_tsne_encoded[:,0], x_train_tsne_encoded[:,1], c=kmeans_labels, cmap="tab20", s=5)
plt.title("t-SNE of Encoded Data with KMeans Clusters")
plt.colorbar()
plt.show()

tsne_dbscan = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
x_train_tsne_dbscan = tsne_dbscan.fit_transform(x_train_encoded)
plt.figure(figsize=(8,6))
plt.scatter(x_train_tsne_dbscan[:,0], x_train_tsne_dbscan[:,1], c=db_scan_labels, cmap="tab20", s=5)
plt.title("t-SNE of Encoded Data with DBSCAN Clusters")
plt.colorbar()
plt.show()

tsne_heirarchical = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
x_train_tsne_heirarchical = tsne_heirarchical.fit_transform(x_train_encoded)
plt.figure(figsize=(8,6))
plt.scatter(x_train_tsne_heirarchical[:,0], x_train_tsne_heirarchical[:,1], c=agg_labels, cmap="tab20", s=5)
plt.title("t-SNE of Encoded Data with Hierarchical Clusters")
plt.colorbar()
plt.show()


def cluster_purity(true_labels, cluster_labels):
    cm = confusion_matrix(true_labels, cluster_labels)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)

purity_kmeans = cluster_purity(y_train_strat, kmeans_labels)
print("K-Means Purity:", purity_kmeans)

purity_dbscan = cluster_purity(y_train_strat, db_scan_labels)
print("DBSCAN Purity:", purity_dbscan)

purity_agg = cluster_purity(y_train_strat, agg_labels)
print("Hierarchical Clustering Purity:", purity_agg)

# use clustering to help train classifier
cluster_train = agg_labels.reshape(-1, 1)
cluster_val = agg.fit_predict(x_val_encoded).reshape(-1, 1)
cluster_test = agg.fit_predict(x_test_encoded).reshape(-1, 1)

X_train_final = np.hstack([x_train_encoded, cluster_train])
X_val_final = np.hstack([x_val_encoded, cluster_val])
X_test_final = np.hstack([x_test_encoded, cluster_test])

X_train_no_cluster = x_train_encoded
X_val_no_cluster   = x_val_encoded
X_test_no_cluster  = x_test_encoded



model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train_encoded.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(20, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train_encoded, y_train_strat,
    validation_data=(x_val_encoded, y_val_strat),
    epochs=30,
    batch_size=32
)

test_loss, test_acc = model.evaluate(x_test_encoded, y_test_strat)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)









