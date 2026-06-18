import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from utils.tensor_utils import denormalize_imagenet

def visualize_pca_of_images(
                            images: tuple, 
                            num_components:int=3
                            ) -> None:
    
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    fig_scale = 4
    
    fig, axes = plt.subplots(
        num_rows, num_cols*2,
        figsize=(num_cols * 2 * fig_scale, num_rows * fig_scale)
    )

    if num_rows == 1:
        axes = axes[np.newaxis, :]

    if num_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (patch_features, image) in enumerate(images):
        row = idx // num_cols
        col = (idx % num_cols) * 2

        # print(images_unmasked.shape)
        h,w = image.shape[1], image.shape[2]

        # print(h,w)
        h , w = h//14, w//14

        pca = PCA(n_components=num_components)
        pca_features = pca.fit_transform(patch_features)

        low_thresh = np.percentile(pca_features, 2)
        high_thresh = np.percentile(pca_features, 98)
        pca_features = np.clip(pca_features, low_thresh, high_thresh)

        rgb_map = (pca_features - low_thresh) / (high_thresh - low_thresh)

        rgb_map = rgb_map.reshape(h,w,3)

        image = denormalize_imagenet(image)

        # print(images_unmasked.shape, rgb_map.shape)

        #orig
        axes[row, col].imshow(image)
        axes[row, col].set_title("Original")
        axes[row, col].axis("off")

        #visualization
        axes[row, col + 1].imshow(rgb_map)
        axes[row, col + 1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_clusters_of_image(
                                images: tuple, 
                                cluster_scores
                                ) -> None:
    
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    fig_scale = 4

    fig, axes = plt.subplots(
        num_rows, num_cols*3,
        figsize=(num_cols * 3 * fig_scale, num_rows * fig_scale)
    )

    if num_rows == 1:
        axes = axes[np.newaxis, :]

    if num_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, ((clusters, features), scores) in enumerate(zip(images, cluster_scores)):
        row = idx // num_cols
        col = (idx % num_cols) * 3
    
        image = features["orig_image"]
        clusters_map_up = clusters["clusters_map_up"]
        num_clusters = clusters["num_clusters"]

        image = denormalize_imagenet(image)
        # print("VIS", image.shape, colored.shape)


        axes[row, col].imshow(image)
        axes[row, col].set_title("Original")
        axes[row, col].axis("off")

        cmap = plt.get_cmap("tab20", num_clusters)
        colored = cmap(clusters_map_up / max(num_clusters - 1, 1))[:, :, :3]

        axes[row, col + 1].imshow(colored)
        axes[row, col + 1].set_title(f"num clusters ({num_clusters})")
        axes[row, col + 1].axis("off")

        legend_handles = []
        for idx, (label_id, score) in enumerate(scores):
            color = cmap(label_id / max(num_clusters - 1, 1))[:3]
            patch = plt.matplotlib.patches.Patch(
                color=color,
                label=f"C{label_id}: {score:.3f}"
            )
            legend_handles.append(patch)

        axes[row, col + 1].legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=7,
            framealpha=0.85,
            title="Cluster : Score",
            title_fontsize=7,
            handlelength=1.2,
            borderpad=0.5
        )

        score_map = np.zeros(clusters_map_up.shape, dtype=np.float32)
        for idx, (label_id, score) in enumerate(scores):
            score_map[clusters_map_up == label_id] = score

        score_min, score_max = score_map.min(), score_map.max()
 
        score_map_normalized = (score_map - score_min) / (score_max - score_min)

        heatmap = plt.get_cmap("RdYlGn")(score_map_normalized)[:, :, :3]   
        blended = np.clip(0.5 * image + 0.5 * heatmap, 0, 1)

        axes[row, col + 2].imshow(blended)
        axes[row, col + 2].set_title("CLIP Score Overlay")
        axes[row, col + 2].axis("off")

        sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=axes[row, col + 2], fraction=0.03, pad=0.04)

    plt.tight_layout()
    plt.show()