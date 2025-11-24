import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from scipy.spatial import cKDTree
import tinycudann as tcnn    

# ============================
# 全域設定
# ============================

MESH_PATH = "chair_base_coacd.obj"  # 你的 base shape mesh

K = 8          # 最近鄰頂點數量 (論文 K=8)
W_CONST = 0.01 # 論文中的 w = 0.01


# ============================
# 載入 base mesh & 建 KD-Tree
# ============================

print(f"[INFO] Loading base mesh from {MESH_PATH} ...")
mesh = trimesh.load(MESH_PATH, process=True)

vertices = mesh.vertices              # (V,3)
vertex_normals = mesh.vertex_normals  # (V,3)

print(f"[INFO] Mesh vertices: {vertices.shape[0]}, faces: {mesh.faces.shape[0]}")

print("[INFO] Building KD-Tree on vertices ...")
kdtree = cKDTree(vertices)


# ============================
# 建每個頂點的 tangent frame T_v
# (簡單版：用 global up 做 Gram-Schmidt)
# ============================

def build_vertex_frames(vertices_np, normals_np):
    """
    為每個頂點建一個簡單的 tangent frame (t, b, n)
    回傳: frames: (V,3,3)，每個 [i,:,:] = [t_i, b_i, n_i]
    """
    V = vertices_np.shape[0]
    frames = np.zeros((V, 3, 3), dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    for i in range(V):
        n = normals_np[i]
        n = n / (np.linalg.norm(n) + 1e-8)

        u = up
        if abs(np.dot(u, n)) > 0.99:
            u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # t = normalize(u - (u·n) n)
        t = u - np.dot(u, n) * n
        t /= (np.linalg.norm(t) + 1e-8)

        # b = n × t
        b = np.cross(n, t)
        b /= (np.linalg.norm(b) + 1e-8)

        frames[i, 0, :] = t
        frames[i, 1, :] = b
        frames[i, 2, :] = n

    return frames

print("[INFO] Building vertex tangent frames ...")
vertex_frames = build_vertex_frames(vertices, vertex_normals)  # (V,3,3)


def query_tangent_frames_numpy(points_np):
    """
    給一組 footpoints x_c，回傳對應的 tangent frame T_c(x)
    points_np: (N,3)
    回傳: frames: (N,3,3)
    """
    # 用同一個 KD-Tree 找最近頂點
    dists, idx = kdtree.query(points_np, k=1)  # idx: (N,)
    frames = vertex_frames[idx]               # (N,3,3)
    return frames


# ============================
# 3.1.2 的 coarse normal n_c(x)
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
    d2 = np.maximum(dists ** 2, eps)
    inv_d2 = 1.0 / d2

    # 頂點法向量的加權和
    term_knn = (n_knn * inv_d2[:, None]).sum(axis=0)  # (3,)

    # 從最近頂點 v1 指向 x 的向量
    v1 = v_knn[0]
    d2_v1 = np.maximum(np.linalg.norm(x - v1) ** 2, eps)
    term_dir = (x - v1) / (W_CONST * d2_v1)

    # 權重總和
    W = inv_d2.sum() + 1.0 / W_CONST

    n_tilde = (term_knn + term_dir) / W
    n = n_tilde / (np.linalg.norm(n_tilde) + 1e-8)

    return n


# ============================
# 3.1.2 的投影 x → x_c, s(x)
# ============================

def project_single_point_numpy(x):
    """
    x: (3,)
    回傳:
      xc: footpoint on base mesh, (3,)
      s : signed distance (float)
      nc: coarse normal n_c(x), (3,)
      miss: bool，此點是否曾經 ray miss 而用了 fallback
    """
    x = np.asarray(x, dtype=np.float64)
    nc = coarse_normal_numpy(x)

    origins = x.reshape(1, 3)
    directions = (-nc).reshape(1, 3)

    # 先假設沒 miss
    miss = False

    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False
    )

    if len(locations) == 0:
        # 射不到就改成朝最近頂點方向射
        miss = True
        print("[WARN] Ray miss, fallback to nearest-vertex direction.")
        dists, idx = kdtree.query(x, k=1)
        v1 = vertices[idx]
        dir_fb = v1 - x
        dir_fb = dir_fb / (np.linalg.norm(dir_fb) + 1e-8)
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=dir_fb.reshape(1, 3),
            multiple_hits=False
        )
        if len(locations) == 0:
            raise RuntimeError("Ray still missed base mesh.")

    xc = locations[0]
    s = float(np.dot(x - xc, nc))  # 向 nc 方向為正

    return xc, s, nc, miss



def project_to_base_numpy(points):
    """
    points: (N,3) numpy
    回傳:
      xc_np: (N,3)
      s_np : (N,)
      nc_np: (N,3)
    並在函式內印出 ray miss 的數量
    """
    xs = []
    ss = []
    ns = []
    miss_count = 0

    for x in points:
        xc, s, nc, miss = project_single_point_numpy(x)
        xs.append(xc)
        ss.append(s)
        ns.append(nc)
        if miss:
            miss_count += 1

    N = len(points)
    print(f"[INFO] Ray miss count: {miss_count} / {N}  "
          f"({miss_count / max(N,1):.2%})")

    return np.stack(xs, axis=0), np.array(ss), np.stack(ns, axis=0)



# ============================
# 3.1.3 Differentiable Projection Layer
# ============================

class ProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        """
        x: (N,3)
        回傳:
          xc: (N,3)
          s : (N,1)
          nc: (N,3)
        """
        device = x.device
        x_np = x.detach().cpu().numpy()

        xc_np, s_np, nc_np = project_to_base_numpy(x_np)

        xc = torch.from_numpy(xc_np).to(device=device, dtype=x.dtype)
        s  = torch.from_numpy(s_np).to(device=device, dtype=x.dtype).unsqueeze(-1)
        nc = torch.from_numpy(nc_np).to(device=device, dtype=x.dtype)

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
            tang = grad_xc - n * dot  # (I - n n^T) grad_xc

            grad_x = tang + n * grad_s  # 加上 ∂s/∂x = n 所貢獻的梯度

        return grad_x  # 只有一個輸入 x


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
        freqs = 2.0 ** torch.arange(
            self.num_frequencies,
            device=s.device,
            dtype=s.dtype
        )  # (F,)

        # (N,F)
        x = s * freqs

        sin = torch.sin(math.pi * x)
        cos = torch.cos(math.pi * x)
        return torch.cat([sin, cos], dim=-1)  # (N, 2F)


class HashGridEncoding(nn.Module):
    """
    tiny-cuda-nn HashGrid encoding 包裝。
    這裡設定：n_levels=16、每層 2 維 → 輸出維度 32，
    等於之前 DummyHashEncoding 的 feat_dim=32。
    """
    def __init__(
        self,
        n_levels=16,
        n_features_per_level=2,
        log2_hashmap_size=19,
        base_resolution=16,
        per_level_scale=1.5,
        bbox_min=None,
        bbox_max=None,
    ):
        super().__init__()

        # tiny-cuda-nn Encoding
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            }
        )
        self.out_dim = n_levels * n_features_per_level

        # 把 base mesh 的 bounding box 存成 buffer，用來把 x_c 正規化到 [0,1]
        if bbox_min is None or bbox_max is None:
            raise ValueError("bbox_min / bbox_max 不能是 None")

        self.register_buffer("bbox_min", torch.from_numpy(bbox_min).float())
        self.register_buffer("bbox_max", torch.from_numpy(bbox_max).float())

    def forward(self, xc):
        """
        xc: (N,3) footpoints (world space)
        會先正規化到 [0,1]^3 再丟進 HashGrid
        """
        # 確保在 GPU + float32
        x = xc

        # [0,1] normalize
        scale = (self.bbox_max - self.bbox_min).clamp(min=1e-6)
        x_norm = (x - self.bbox_min) / scale
        x_norm = x_norm.contiguous()  # tcnn 要求 contiguous

        return self.encoding(x_norm)



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

        # 共 10 維：sigma, kd(3), ks(3), gloss, theta, phi
        self.out_layer = nn.Linear(hidden_dim, 10)

    def forward(self, feat, sdf_embed):
        x = torch.cat([feat, sdf_embed], dim=-1)  # (N, in_dim)
        h = self.mlp(x)
        out = self.out_layer(h)

        sigma = out[..., 0:1]
        kd    = out[..., 1:4]
        ks    = out[..., 4:7]
        gloss = out[..., 7:8]
        theta = out[..., 8:9]
        phi   = out[..., 9:10]
        return sigma, kd, ks, gloss, theta, phi

class FineNormalHead(nn.Module):
    """用 f_hat(x) 預測 fine normal residual（在 tangent frame 裡）"""
    def __init__(self, feat_dim=32, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),   # 輸出 (N,3) 的向量
        )

    def forward(self, feat_hat):
        # 用 tanh 把幅度卡住，不要太爆
        n_res_local = torch.tanh(self.mlp(feat_hat))
        return n_res_local  # (N,3)


class NeRFTextureAttributes(nn.Module):
    """
    3.1.4 + 真正 tiny-cuda-nn HashGrid：
      - f(x): HashGridEncoding
      - \hat{f}(x): 另一個 HashGridEncoding（獨立參數）
    """
    def __init__(self, sdf_freqs=6, bbox_min=None, bbox_max=None):
        super().__init__()

        if bbox_min is None or bbox_max is None:
            raise ValueError("NeRFTextureAttributes 需要 bbox_min / bbox_max")

        # f(x) 的 HashGrid
        self.hash_f = HashGridEncoding(
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=16,
            per_level_scale=1.5,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        # \hat{f}(x) 的 HashGrid（獨立參數）
        self.hash_fh = HashGridEncoding(
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=16,
            per_level_scale=1.5,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

        feat_dim = self.hash_f.out_dim  # 32

        self.sdf_enc  = SDFEncoding(num_frequencies=sdf_freqs)
        self.mlp_main = AttributeMLP(
            feat_dim=feat_dim,
            sdf_embed_dim=2 * sdf_freqs
        )

        # 用 f_hat 做 fine normal residual
        self.fine_normal_head = FineNormalHead(
            feat_dim=feat_dim,
            hidden_dim=64
        )

    def forward(self, xc, s):
        """
        xc: (N,3)  footpoint
        s : (N,1)  signed distance
        回傳:
          sigma, kd, ks, g, theta, phi, f, f_hat, n_res_local
        """
        f     = self.hash_f(xc)      # (N,feat_dim)
        f_hat = self.hash_fh(xc)     # (N,feat_dim)
        sdf_embed = self.sdf_enc(s)

        sigma, kd, ks, g, theta, phi = self.mlp_main(f, sdf_embed)
        n_res_local = self.fine_normal_head(f_hat)      # (N,3)

        return sigma, kd, ks, g, theta, phi, f, f_hat, n_res_local



# ============================
# 3.1.5 Shading（簡化 Phong）
# ============================

def local_angles_to_normal(theta, phi):
    """
    theta, phi: (N,1)
    回傳: n_local: (N,3)
    """
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    sin_p = torch.sin(phi)
    cos_p = torch.cos(phi)

    x = sin_t * cos_p
    y = sin_t * sin_p
    z = cos_t
    return torch.cat([x, y, z], dim=-1)


def phong_shading(kd, ks, gloss, n_f, light_dir, view_dir):
    """
    kd, ks: (N,3)
    gloss: (N,1)
    n_f:   (N,3)
    light_dir, view_dir: (3,)
    回傳: rgb: (N,3)
    """
    L = light_dir / (light_dir.norm() + 1e-8)
    V = view_dir  / (view_dir.norm() + 1e-8)

    L = L.view(1, 3).expand_as(n_f)
    V = V.view(1, 3).expand_as(n_f)

    ndotl = torch.clamp((n_f * L).sum(dim=-1, keepdim=True), min=0.0)
    diffuse = kd * ndotl

    R = 2 * ndotl * n_f - L
    rdotv = torch.clamp((R * V).sum(dim=-1, keepdim=True), min=0.0)
    specular = ks * (rdotv ** torch.clamp(gloss, min=1.0))

    color = diffuse + specular
    color = torch.clamp(color, 0.0, 1.0)
    return color


# ============================
# Demo：整條 3.1.3 → 3.1.5 forward + backward
# ============================

def main():
    # 有 GPU 就用 cuda，沒有就用 cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. base mesh 的 bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    # 2. Projection layer + Attribute 網路
    proj_layer = DifferentiableProjectionLayer().to(device)
    attr_net   = NeRFTextureAttributes(
        sdf_freqs=6,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    ).to(device)

    # 3. 在 bounding box 裡取 N 個 random points 當 sample
    N = 256
    pts_np = bbox_min + np.random.rand(N, 3) * (bbox_max - bbox_min)
    x = torch.from_numpy(pts_np).float().to(device)
    x.requires_grad_(True)

    # 4. differentiable projection: x → (x_c, s, n_c)
    print("[INFO] Running differentiable projection ...")
    xc, s, nc = proj_layer(x)  # xc:(N,3), s:(N,1), nc:(N,3)

    # 5. 查 tangent frame T_c(x)（numpy + KDTree，當常數使用）
    xc_np = xc.detach().cpu().numpy()
    Tc_np = query_tangent_frames_numpy(xc_np)      # (N,3,3)
    Tc = torch.from_numpy(Tc_np).to(device=device, dtype=x.dtype)  # (N,3,3)

    # 6. Attributes prediction（含 f, f_hat, fine normal residual）
    print("[INFO] Running attributes prediction ...")
    sigma, kd, ks, g, theta, phi, f, f_hat, n_res_local = attr_net(xc, s)

    # 7. coarse normal（由 theta, phi 決定）
    n_local = local_angles_to_normal(theta, phi)        # (N,3) ，tangent frame 下
    n_local_3d = n_local.unsqueeze(-1)                  # (N,3,1)
    n_coarse = torch.matmul(Tc, n_local_3d).squeeze(-1) # (N,3)，world space
    n_coarse = n_coarse / (torch.norm(n_coarse, dim=-1, keepdim=True) + 1e-8)

    # 8. fine normal residual：先在 tangent frame，轉到 world 再相加
    n_res_local_3d = n_res_local.unsqueeze(-1)          # (N,3,1)
    n_res_world = torch.matmul(Tc, n_res_local_3d).squeeze(-1)  # (N,3)

    n_final = n_coarse + n_res_world
    n_final = n_final / (torch.norm(n_final, dim=-1, keepdim=True) + 1e-8)

    # 9. Shading
    light_dir = torch.tensor([0.5, 0.8, 0.2], device=device, dtype=x.dtype)
    view_dir  = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=x.dtype)

    print("[INFO] Running shading ...")
    rgb = phong_shading(kd, ks, g, n_final, light_dir, view_dir)  # (N,3)

    # 10. Loss + backward（純測試用）
    rgb_gt = torch.zeros_like(rgb)

    loss_color = ((rgb - rgb_gt) ** 2).mean()
    loss_reg = (
        sigma.pow(2).mean()
        + kd.pow(2).mean()
        + ks.pow(2).mean()
        + g.pow(2).mean()
        + theta.pow(2).mean()
        + phi.pow(2).mean()
    )
    loss = loss_color + 1e-3 * loss_reg

    print("[INFO] Loss_color:", float(loss_color.item()))
    print("[INFO] Loss_reg  :", float(loss_reg.item()))
    print("[INFO] Total loss:", float(loss.item()))

    loss.backward()
    grad_norm = x.grad.norm().item()
    print("[INFO] Grad norm on x:", grad_norm)


if __name__ == "__main__":
    main()
