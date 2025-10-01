from torchvision import transforms
from sklearn.decomposition import PCA
from scipy import signal


def make_transform(size_h, size_w, patch_size):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        (size_h * patch_size, size_w * patch_size), 
        antialias=True
        )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


def get_input_sizes(image, num_tokens=3600):
    original_width, original_height = image.size[-2:]
    aspect_ratio = original_width / original_height
    base_h, base_w = int((num_tokens / aspect_ratio) ** 0.5), int((num_tokens * aspect_ratio) ** 0.5)
    return base_h, base_w


def do_pca(feats):
    h, w = feats.shape[-2:]
    x = feats.flatten(1).permute(1, 0)

    pca = PCA(n_components=3, whiten=True)
    features_pca = pca.fit_transform(x.numpy())
    pca_image = features_pca.reshape((h, w, 3))

    pca_image_normalized = (pca_image - pca_image.min(axis=(0, 1), keepdims=True))
    pca_image_normalized /= (pca_image.max(axis=(0, 1), keepdims=True) - pca_image.min(axis=(0, 1), keepdims=True))
    return pca_image_normalized 
