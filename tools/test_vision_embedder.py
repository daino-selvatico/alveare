import numpy as np
from PIL import Image
from pathlib import Path

def test_vision_embedder(image_path, weights_dir="quantized_weights_gemma4/vision"):
    weights_dir = Path(weights_dir)
    print(f"Loading vision weights from {weights_dir}...")
    
    patch_norm1_w = np.load(weights_dir / "v_patch_norm_1_weight.npy")
    patch_norm1_b = np.load(weights_dir / "v_patch_norm_1_bias.npy")
    patch_embd_w = np.load(weights_dir / "v_patch_embd_weight.npy") # (3840, 6912)
    patch_embd_b = np.load(weights_dir / "v_patch_embd_bias.npy")   # (3840,)
    patch_norm2_w = np.load(weights_dir / "v_patch_norm_2_weight.npy")
    patch_norm2_b = np.load(weights_dir / "v_patch_norm_2_bias.npy")
    pos_embd = np.load(weights_dir / "v_position_embd_weight.npy")  # (2, 1120, 3840)
    patch_norm3_w = np.load(weights_dir / "v_patch_norm_3_weight.npy")
    patch_norm3_b = np.load(weights_dir / "v_patch_norm_3_bias.npy")
    mm_proj_w = np.load(weights_dir / "mm_input_projection_weight.npy") # (3840, 3840)
    
    print(f"Loading image from {image_path}...")
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    print(f"Original image dimensions: {W}x{H}")
    
    # Pad / Resize to multiples of 48
    target_W = ((W + 47) // 48) * 48
    target_H = ((H + 47) // 48) * 48
    # Cap max dimensions to e.g. 10x10 = 100 patches (480x480) or similar
    if target_W > 480 or target_H > 480:
        ratio = min(480 / W, 480 / H)
        new_w = max(48, int(round((W * ratio) / 48) * 48))
        new_h = max(48, int(round((H * ratio) / 48) * 48))
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        W, H = img.size
        target_W, target_H = W, H
        
    print(f"Resized image dimensions: {target_W}x{target_H}")
    img_arr = np.array(img, dtype=np.float32) / 255.0 # normalize [0, 1]
    
    num_patches_x = target_W // 48
    num_patches_y = target_H // 48
    total_patches = num_patches_x * num_patches_y
    print(f"Grid: {num_patches_x} x {num_patches_y} = {total_patches} patches")
    
    # Extract patches: each patch is 48x48x3 = 6912 floats
    patches = []
    positions = []
    for py in range(num_patches_y):
        for px in range(num_patches_x):
            patch = img_arr[py*48:(py+1)*48, px*48:(px+1)*48, :]
            patches.append(patch.flatten())
            positions.append((px, py))
            
    patches = np.array(patches, dtype=np.float32) # (N, 6912)
    
    # LayerNorm 1 on raw patches
    mean1 = patches.mean(axis=-1, keepdims=True)
    var1 = patches.var(axis=-1, keepdims=True)
    norm1 = (patches - mean1) / np.sqrt(var1 + 1e-6)
    norm1 = norm1 * patch_norm1_w + patch_norm1_b
    
    # Linear projection: (N, 6912) @ (6912, 3840) + bias -> (N, 3840)
    emb = norm1 @ patch_embd_w.T + patch_embd_b
    
    # LayerNorm 2
    mean2 = emb.mean(axis=-1, keepdims=True)
    var2 = emb.var(axis=-1, keepdims=True)
    norm2 = (emb - mean2) / np.sqrt(var2 + 1e-6)
    norm2 = norm2 * patch_norm2_w + patch_norm2_b
    
    # Add 2D factorized positional embeddings
    for i, (px, py) in enumerate(positions):
        norm2[i] += pos_embd[0, px, :] + pos_embd[1, py, :]
        
    # LayerNorm 3
    mean3 = norm2.mean(axis=-1, keepdims=True)
    var3 = norm2.var(axis=-1, keepdims=True)
    norm3 = (norm2 - mean3) / np.sqrt(var3 + 1e-6)
    norm3 = norm3 * patch_norm3_w + patch_norm3_b
    
    # Final Multimodal projection: (N, 3840) @ (3840, 3840).T -> (N, 3840)
    final_tokens = norm3 @ mm_proj_w.T
    
    print(f"Generated {final_tokens.shape[0]} visual token embeddings of dimension {final_tokens.shape[1]}")
    print(f"Embedding stats: min={final_tokens.min():.4f}, max={final_tokens.max():.4f}, mean={final_tokens.mean():.4f}, std={final_tokens.std():.4f}")
    return final_tokens

if __name__ == "__main__":
    test_vision_embedder("/home/daino/.gemini/antigravity-cli/brain/0986b6ff-695c-4dfa-9b2b-60ff64b70271/.user_uploaded/uploaded_media_1787589942586.png")
