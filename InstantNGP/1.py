import sys, time
sys.path.append(r"C:\Users\s7103\instant-ngp\build")

import pyngp as ngp

# 建立 NeRF Testbed
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)

# 載入訓練資料
testbed.load_training_data("C:/Users/s7103/instant-ngp/data/nerf/fox")

# ✅ 指定預設設定檔（位於 instant-ngp/configs/nerf/base.json）
testbed.reload_network_from_file("C:/Users/s7103/instant-ngp/configs/nerf/base.json")

# 訓練迴圈

while True:
    testbed.frame()
    loss = testbed.loss  # 修正：不呼叫，直接取得屬性
    print(f"Loss = {loss:.6f}")
    if loss < 0.001:
        print("✅ Loss reached threshold. Stopping training.")
        testbed.save_snapshot(
            "C:/Users/s7103/instant-ngp/data/nerf/fox/snapshot.msgpack", False
        )
        break
    time.sleep(0.01)
#將訓練結果的圖片呈現出來
# ...existing code...
#將訓練結果的圖片呈現出來
# 嘗試多種可能的屬性/方法名稱以相容不同版本的 pyngp 綁定
try:
    if hasattr(testbed, "set_nerf_render_mode"):
        testbed.set_nerf_render_mode(ngp.NerfRenderMode.Fancy)
    elif hasattr(testbed, "nerf_render_mode"):
        testbed.nerf_render_mode = ngp.NerfRenderMode.Fancy
    elif hasattr(testbed, "nerf") and hasattr(testbed.nerf, "render_mode"):
        testbed.nerf.render_mode = ngp.NerfRenderMode.Fancy
    else:
        print("⚠️ 無法設定 Nerf render mode：Testbed API 不支援可設定的屬性/方法。")
except Exception as e:
    print(f"⚠️ 設定 render mode 時發生例外: {e}")

# ...existing code...