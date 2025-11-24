import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from scipy.spatial import cKDTree

# ============================
# 全域設定
# ============================

# 請確認這個檔案在同一個資料夾
MESH_PATH = "chair_base_coacd.obj"

K = 8          # 最近鄰頂點數 (論文 K=8)
W_CONST = 0.01 # 論文裡的 w = 0.01


# ============================
# 載入 base mesh, 建 KD-Tree
# ============================

print(f"[INFO] Loading base mesh from {MESH_PATH} ...")
mesh = trimesh.load(MESH_PATH, process=True)

vertices = mesh.vertices              # (V,3)
vertex_normals = mesh.vertex_normals  # (V,3)

print(f"[INFO] Mesh vertices: {vertices.shape[0]}, faces: {mesh.faces.shape[0]}")

print("[INFO] Building KD-Tree on vertices ...")
kdtree = cKDTree(vertices)


# ============================
# 粗法向量 n_c(x)：3.1.2 式 (1)
# ============================

def coarse_normal_numpy(x):
    """
    x: (3,) numpy array
    return: 單位長度的 n_c(x), shape (3,)
    """
    x = np.asarray(x, dtype=np.float64)

    # 找 K 個最近頂點
    dists, idx = kdtree.query(x, k=K)  # dists: (K,), idx: (K,)
    v_knn = vertices[idx]              # (K,3)
    n_knn = vertex_normals[idx]        # (K,3)

    eps = 1e-8
    d2 = np.maximum(dists ** 2, eps)   # 避免除以 0
    inv_d2 = 1.0 / d2                  # (K,)

    # 第一項：K 個頂點法向量的加權平均
    term_knn = (n_knn * inv_d2[:, None]).sum(axis=0)  # (3,)

    # 第二項：從最近頂點 v1 指向 x 的方向
    v1 = v_knn[0]
    d2_v1 = np.maximum(np.linalg.norm(x - v1) ** 2, eps)
    term_dir = (x - v1) / (W_CONST * d2_v1)

    # 權重總和 W
    W = inv_d2.sum() + 1.0 / W_CONST

    n_tilde = (term_knn + term_dir) / W
    n = n_tilde / (np.linalg.norm(n_tilde) + 1e-8)

    return n


# ============================
# ray 投影 x → x_c, s(x)
# ============================

def project_single_point_numpy(x):
    """
    x: (3,)
    回傳:
      xc: footpoint on base mesh, (3,)
      s : signed distance (float)
      nc: coarse normal n_c(x), (3,)
    """
    x = np.asarray(x, dtype=np.float64)
    nc = coarse_normal_numpy(x)

    origins = x.reshape(1, 3)
    directions = (-nc).reshape(1, 3)

    # 用 trimesh 的射線求交 (需要 rtree)
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    if len(locations) == 0:
        # 理論上不太常發生，保險起見 fallback 射向最近頂點
        print("[WARN] Ray miss, fallback to nearest-vertex direction.")
        dists, idx = kdtree.query(x, k=1)
        v1 = vertices[idx]
        dir_fb = v1 - x
        dir_fb = dir_fb / (np.linalg.norm(dir_fb) + 1e-8)
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dir_fb.reshape(1, 3),
            multiple_hits=False
        )
        if len(locations) == 0:
            raise RuntimeError("Ray still missed base mesh.")

    xc = locations[0]
    s = float(np.dot(x - xc, nc))  # 向 nc 方向為正

    return xc, s, nc


def project_to_base_numpy(points):
    """
    points: (N,3) numpy
    回傳:
      xc_np: (N,3)
      s_np : (N,)
      nc_np: (N,3)
    """
    xs = []
    ss = []
    ns = []
    for x in points:
        xc, s, nc = project_single_point_numpy(x)
        xs.append(xc)
        ss.append(s)
        ns.append(nc)
    return np.stack(xs, axis=0), np.array(ss), np.stack(ns, axis=0)


# ============================
# 3.1.3 Differentiable Projection Layer
# ============================

class ProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        x: (N,3) float tensor (CPU or GPU)
        回傳:
          xc: (N,3)
          s : (N,1)
          nc: (N,3)
        """
        device = x.device
        x_np = x.detach().cpu().numpy()  # (N,3)

        xc_np, s_np, nc_np = project_to_base_numpy(x_np)

        xc = torch.from_numpy(xc_np).to(device=device, dtype=x.dtype)
        s  = torch.from_numpy(s_np).to(device=device, dtype=x.dtype).unsqueeze(-1)
        nc = torch.from_numpy(nc_np).to(device=device, dtype=x.dtype)

        # backward 只需要 n_c(x)
        ctx.save_for_backward(nc)

        return xc, s, nc

    @staticmethod
    def backward(ctx, grad_xc, grad_s, grad_nc):
        """
        根據論文 3.1.3 式 (2)：
          ∂x_c/∂x = I - n n^T
          ∂s/∂x   = n
        """
        (nc,) = ctx.saved_tensors  # (N,3)
        grad_x = None

        if ctx.needs_input_grad[0]:
            n = nc  # (N,3)

            # (n · grad_xc) → (N,1)
            dot = (grad_xc * n).sum(dim=-1, keepdim=True)
            # (I - n n^T) grad_xc = grad_xc - n (n·grad_xc)
            tang = grad_xc - n * dot

            # ∂s/∂x = n → contribution from grad_s
            grad_x = tang + n * grad_s  # (N,3)

        # 只有第一個輸入 x 有梯度；其它回傳 None
        return grad_x, None, None


class DifferentiableProjectionLayer(nn.Module):
    def forward(self, x):
        return ProjectionFn.apply(x)


# ============================
# 3.1.4 Attributes Prediction
# ============================

class SDFEncoding(nn.Module):
    """對 scalar s 做 Fourier encoding."""
    def __init__(self, num_frequencies=6):
        super().__init__()
        self.num_frequencies = num_frequencies

    def forward(self, s):
        """
        s: (N,1)
        回傳: (N, 2 * num_frequencies)
        """
        # freqs: [1, 2, 4, ...]
        freqs = 2.0 ** torch.arange(
            self.num_frequencies,
            device=s.device,
            dtype=s.dtype
        )  # (F,)

        # (N,1,F)
        x = s.unsqueeze(-1) * freqs  # broadcast

        # sin, cos
        sin = torch.sin(math.pi * x)
        cos = torch.cos(math.pi * x)
        # (N, 2F)
        return torch.cat([sin, cos], dim=-1)


class DummyHashEncoding(nn.Module):
    """
    用簡單 MLP 代替 hash grid encoding。
    之後你可以換成 tiny-cuda-nn 的 HashGridEncoding。
    """
    def __init__(self, in_dim=3, out_dim=32, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, xc):
        """
        xc: (N,3) footpoints on base mesh
        """
        return self.net(xc)


class AttributeMLP(nn.Module):
    """
    根據 f(x) + s(x) 的 encoding 預測：
      sigma, kd(3), ks(3), gloss, theta, phi
    """
    def __init__(self, feat_dim=32, sdf_embed_dim=12, hidden_dim=64):
        super().__init__()
        in_dim = feat_dim + sdf_embed_dim

        layers = []
        for i in range(3):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

        # 輸出共 10 維：1+3+3+1+1+1
        self.out_layer = nn.Linear(hidden_dim, 10)

    def forward(self, feat, sdf_embed):
        """
        feat: (N,feat_dim)
        sdf_embed: (N,sdf_embed_dim)
        """
        x = torch.cat([feat, sdf_embed], dim=-1)
        h = self.mlp(x)
        out = self.out_layer(h)

        sigma = out[..., 0:1]  # (N,1)
        kd    = out[..., 1:4]  # (N,3)
        ks    = out[..., 4:7]  # (N,3)
        gloss = out[..., 7:8]  # (N,1)
        theta = out[..., 8:9]  # (N,1)
        phi   = out[..., 9:10] # (N,1)

        return sigma, kd, ks, gloss, theta, phi


class NeRFTextureAttributes(nn.Module):
    """
    3.1.4 的簡化版：
      - 用 f(x) (hash-ish) + s(x) encoding 做主要屬性
      - 用 \hat{f}(x) 額外做 fine normal 的部分（這裡先回傳給你後續用）
    """
    def __init__(self, feat_dim=32, sdf_freqs=6):
        super().__init__()
        self.hash_f   = DummyHashEncoding(out_dim=feat_dim)
        self.hash_fh  = DummyHashEncoding(out_dim=feat_dim)
        self.sdf_enc  = SDFEncoding(num_frequencies=sdf_freqs)
        self.mlp_main = AttributeMLP(
            feat_dim=feat_dim,
            sdf_embed_dim=2 * sdf_freqs
        )

    def forward(self, xc, s):
        """
        xc: (N,3)  footpoint
        s : (N,1)  signed distance
        回傳:
          sigma, kd, ks, g, theta, phi, f, f_hat
        """
        f     = self.hash_f(xc)
        f_hat = self.hash_fh(xc)

        sdf_embed = self.sdf_enc(s)

        sigma, kd, ks, g, theta, phi = self.mlp_main(f, sdf_embed)

        return sigma, kd, ks, g, theta, phi, f, f_hat


# ============================
# Demo：隨機 sample 一些點，跑 forward/backward
# ============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 建 projection layer + attribute network
    proj_layer = DifferentiableProjectionLayer().to(device)
    attr_net   = NeRFTextureAttributes(feat_dim=32, sdf_freqs=6).to(device)

    # 在 mesh bounding box 裡取 N 個隨機點
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    N = 256

    pts_np = bbox_min + np.random.rand(N, 3) * (bbox_max - bbox_min)
    x = torch.from_numpy(pts_np).float().to(device)
    x.requires_grad_(True)

    print("[INFO] Running differentiable projection ...")
    xc, s, nc = proj_layer(x)  # xc: (N,3), s: (N,1), nc: (N,3)

    print("[INFO] Running attributes prediction ...")
    sigma, kd, ks, g, theta, phi, f, f_hat = attr_net(xc, s)

    # 做一個簡單 loss（只是測試 backward；實際訓練會用顏色 MSE）
    loss = (
        sigma.pow(2).mean()
        + kd.pow(2).mean()
        + ks.pow(2).mean()
        + g.pow(2).mean()
        + theta.pow(2).mean()
        + phi.pow(2).mean()
    )

    print("[INFO] Loss:", float(loss.item()))
    loss.backward()

    # 看一下 x 的梯度是不是有被回傳
    grad_norm = x.grad.norm().item()
    print("[INFO] Grad norm on x:", grad_norm)


if __name__ == "__main__":
    main()
