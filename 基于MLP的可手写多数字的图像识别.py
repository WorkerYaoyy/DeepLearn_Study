"""
手写数字识别 — 鼠标绘制 + 实时识别
在画布上用鼠标画数字，按空格键识别，按 C 清除
"""

import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免负号显示为方块
import matplotlib.pyplot as plt


# ============================================================
# 1. 模型定义
# ============================================================
class MnistModel(nn.Module):
    """三层全连接网络: 784 → 256 → ReLU → 128 → ReLU → 10"""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


# ============================================================
# 2. 数据准备
# ============================================================
def img_preprocess(img) -> Tensor:
    """PIL Image → 展平 → 归一化到 [0, 1]"""
    tensor = transforms.functional.to_tensor(img)
    return tensor.view(28 * 28)

train_dataset = torchvision.datasets.MNIST(
    root="./dataset", train=True, download=True, transform=img_preprocess,
)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)


# ============================================================
# 3. 模型加载 / 训练（缓存判断）
# ============================================================
MODEL_PATH = "Training_dataset/mnist_model.pth"
model = MnistModel()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"✅ 已加载已有模型 → {MODEL_PATH}")
else:
    print("⚠️ 未找到模型，开始训练...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
        print(f"  Epoch {epoch+1}/{epochs} — loss: {total_loss/total:.4f}, acc: {correct/total:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ 训练完成 → {MODEL_PATH}")


# ============================================================
# 4. 手写识别 GUI
# ============================================================
class HandwritingRecognizer:
    def __init__(self, model, canvas_size=280, pen_width=15, eraser_width=30):
        self.model = model
        self.model.eval()
        self.canvas_size = canvas_size
        self.pen_width = pen_width
        self.eraser_width = eraser_width
        self._last_x = None
        self._last_y = None

        # ★ 内存 PIL 图像：同步绘制，消除屏幕截图的坐标偏移问题
        self._pil_img = Image.new("L", (canvas_size, canvas_size), 255)  # 白底
        self._draw = ImageDraw.Draw(self._pil_img)

        # 主窗口
        self.window = tk.Tk()
        self.window.title("手写数字识别 — 左键画 | 右键擦 | 空格识别 | C 清空")

        # 画布（白底）
        self.canvas = tk.Canvas(
            self.window, width=canvas_size, height=canvas_size,
            bg="white", cursor="cross",
        )
        self.canvas.pack(padx=10, pady=10)

        # 结果标签
        self.result_label = tk.Label(
            self.window,
            text="✏️ 左键画数字  |  🧹 右键擦除  |  空格识别  |  C 清空",
            font=("Microsoft YaHei", 14),
        )
        self.result_label.pack(pady=5)

        # 按钮
        btn_frame = tk.Frame(self.window)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="识别 (空格)", command=self.predict,
                  font=("Microsoft YaHei", 12)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="清空 (C)", command=self.clear,
                  font=("Microsoft YaHei", 12)).pack(side="left", padx=5)

        # 事件绑定
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<Button-2>", self._on_right_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.window.bind("<space>", lambda e: self.predict())
        self.window.bind("<c>", lambda e: self.clear())
        self.window.bind("<C>", lambda e: self.clear())

    # ---------- 绘画（左键） ----------
    def _on_left_click(self, event):
        self._last_x, self._last_y = event.x, event.y
        self._draw_on_both(event.x, event.y, self.pen_width, fill=0)     # 0=黑色(PIL)

    def _on_left_drag(self, event):
        self._draw_line_on_both(event.x, event.y, self.pen_width, fill=0)

    # ---------- 擦除（右键） ----------
    def _on_right_click(self, event):
        self._last_x, self._last_y = event.x, event.y
        self._draw_on_both(event.x, event.y, self.eraser_width, fill=255)  # 255=白色

    def _on_right_drag(self, event):
        self._draw_line_on_both(event.x, event.y, self.eraser_width, fill=255)

    def _on_release(self, event):
        self._last_x, self._last_y = None, None

    # ---------- 双通道绘制：tkinter 画布 + PIL 内存图 ----------
    def _draw_on_both(self, x, y, size, fill):
        """在 (x, y) 画点，同时绘制到 canvas 和 PIL Image"""
        r = size // 2
        # tkinter 画布（视觉反馈）
        color = "black" if fill == 0 else "white"
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r, fill=color, outline=color,
        )
        # PIL 内存图（精确捕获）
        self._draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def _draw_line_on_both(self, x, y, size, fill):
        """从上一个点连线到 (x, y)，双通道绘制"""
        r = size // 2
        color = "black" if fill == 0 else "white"

        if self._last_x is not None and self._last_y is not None:
            # tkinter
            self.canvas.create_line(
                self._last_x, self._last_y, x, y,
                width=size, fill=color, capstyle=tk.ROUND, joinstyle=tk.ROUND,
            )
            # PIL（两端再补圆头，模拟 tkinter ROUND capstyle）
            self._draw.line(
                [self._last_x, self._last_y, x, y], fill=fill, width=size,
            )
            self._draw.ellipse(
                [x - r, y - r, x + r, y + r], fill=fill,
            )
        else:
            self._draw_on_both(x, y, size, fill)

        self._last_x, self._last_y = x, y

    def clear(self):
        """清空画布 + PIL 图像"""
        self.canvas.delete("all")
        self._pil_img = Image.new("L", (self.canvas_size, self.canvas_size), 255)
        self._draw = ImageDraw.Draw(self._pil_img)

    # ---------- 获取图像 ----------
    def get_image(self) -> Image.Image:
        """直接返回内存 PIL 图像，无需截屏"""
        return self._pil_img.copy()

    # ---------- 多数字分割 ----------
    def _find_digits(self, img: Image.Image):
        """
        连通区域分析：找出图中每个独立数字的边界框
        返回按从左到右排序的 [(x1, y1, x2, y2), ...] 和每块裁剪图
        """
        arr = np.array(img)                                 # [H, W], 0=黑色笔画, 255=白背景
        binary = (arr < 128).astype(np.uint8)               # 笔画=1, 背景=0

        H, W = binary.shape
        visited = np.zeros((H, W), dtype=bool)
        regions = []
        min_pixels = 30                                     # 忽略太小的噪点

        for y in range(H):
            for x in range(W):
                if binary[y, x] and not visited[y, x]:
                    # BFS 搜连通区域
                    queue = [(y, x)]
                    visited[y, x] = True
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    count = 0

                    while queue:
                        cy, cx = queue.pop(0)
                        count += 1
                        min_x, max_x = min(min_x, cx), max(max_x, cx)
                        min_y, max_y = min(min_y, cy), max(max_y, cy)
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < H and 0 <= nx < W:
                                    if binary[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        queue.append((ny, nx))

                    if count >= min_pixels:
                        regions.append((min_x, min_y, max_x, max_y))

        # 按 x 坐标从左到右排序（阅读顺序）
        regions.sort(key=lambda b: b[0])

        digits = []
        for (x1, y1, x2, y2) in regions:
            # 加一点 padding，防笔画贴边被截断
            pad = 4
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(W - 1, x2 + pad)
            y2 = min(H - 1, y2 + pad)
            crop = img.crop((x1, y1, x2, y2))
            digits.append(((x1, y1, x2, y2), crop))

        return digits

    def preprocess(self, img: Image.Image, shift_x=0, shift_y=0, scale=1.0) -> Tensor:
        """裁剪图 → 保持长宽比 → 居中 → 28×28 → 归一化 → 展平 → 反转
        shift_x/y: 平移增强 (±像素), scale: 缩放增强 (1.0=不变)"""
        w, h = img.size
        # 按 scale 缩放后的大小
        new_w, new_h = int(w * scale), int(h * scale)
        if scale != 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)

        # 正方形画布（边长取长边），居中 + shift 偏移
        side = max(new_w, new_h)
        canvas = Image.new("L", (side, side), 255)
        offset_x = (side - new_w) // 2 + shift_x
        offset_y = (side - new_h) // 2 + shift_y
        canvas.paste(img, (offset_x, offset_y))

        # 缩放到 20×20，放回 28×28 居中
        scale_size = 20
        padded = Image.new("L", (28, 28), 255)
        resized = canvas.resize((scale_size, scale_size), Image.LANCZOS)
        padded.paste(resized, ((28 - scale_size) // 2, (28 - scale_size) // 2))

        tensor = transforms.functional.to_tensor(padded)
        tensor = 1.0 - tensor
        return tensor.view(28 * 28)

    # ---------- 识别 ----------
    def predict(self):
        """分割多个数字 → 逐个识别 → 合并结果显示"""
        img_raw = self.get_image()
        digits = self._find_digits(img_raw)                 # 连通区域分割

        if not digits:
            self.result_label.config(
                text="⚠️ 未检测到数字，请在画布上画一个数字",
                fg="red",
            )
            return

        # TTA 增强组合：原图 + 上下左右偏移 + 缩放变化 → 投票
        tta_shifts = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)]
        tta_scales = [1.0, 0.95, 1.05]

        results = []
        tensors = []
        for bbox, crop in digits:
            # 对每个数字，生成 TTA 变体并取平均 logits
            avg_probs = None
            count = 0
            for sx, sy in tta_shifts:
                for sc in tta_scales:
                    tensor = self.preprocess(crop, shift_x=sx, shift_y=sy, scale=sc)
                    with torch.no_grad():
                        logits = self.model(tensor.unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)
                    avg_probs = probs if avg_probs is None else avg_probs + probs
                    count += 1

            avg_probs = avg_probs / count                          # 平均概率
            pred = avg_probs.argmax(dim=1).item()
            confidence = avg_probs[0, pred].item()

            # 保存原始（不打增强的）tensor 用于预览
            base_tensor = self.preprocess(crop)
            tensors.append((bbox, crop, base_tensor))
            results.append((pred, confidence))

        # 合并显示
        digits_str = "".join(str(p) for p, _ in results)
        avg_conf = sum(c for _, c in results) / len(results)
        self.result_label.config(
            text=f"识别结果: {digits_str}   (置信度: {avg_conf:.1%})",
            fg="green" if avg_conf > 0.8 else "orange",
        )

        # 调试预览
        self._show_preview(img_raw, digits, tensors, results)

    def _show_preview(self, img_raw, digits, tensors, results):
        """预览：原始图+边界框 + 每个数字的 28×28 模型输入"""
        n = len(results)
        if n == 0:
            return

        fig, axes = plt.subplots(2, max(n, 1), figsize=(max(n, 1) * 2.5, 5))
        if n == 1:
            axes = axes.reshape(2, 1)  # 统一形状，方便用 axes[r,c] 索引

        digits_str = "".join(str(p) for p, _ in results)
        fig.suptitle(f"检测到 {n} 个数字: {digits_str}", fontsize=14, color="blue")

        # ===== 第一行：原始图 + 边界框 =====
        overlay = img_raw.copy().convert("RGB")
        draw_overlay = ImageDraw.Draw(overlay)
        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        for i, ((x1, y1, x2, y2), _) in enumerate(digits):
            color = colors[i % len(colors)]
            draw_overlay.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw_overlay.text((x1, y1 - 10), f"#{i+1}", fill=color)

        axes[0, 0].imshow(overlay)
        axes[0, 0].set_title(f"分割: {n} 个数字")
        axes[0, 0].axis("off")
        for col in range(1, n):
            axes[0, col].axis("off")

        # ===== 第二行：每个数字的 28×28 模型输入 =====
        for i, ((bbox, crop, tensor), (pred, conf)) in enumerate(zip(tensors, results)):
            img_28 = tensor.view(28, 28).numpy()
            axes[1, i].imshow(img_28, cmap="gray")
            color = "green" if conf > 0.8 else "red"
            axes[1, i].set_title(f"#{i+1}: {pred} ({conf:.1%})", color=color)
            axes[1, i].axis("off")

        plt.tight_layout()
        plt.show()

    def run(self):
        self.window.mainloop()


# ============================================================
# 5. 启动
# ============================================================
if __name__ == "__main__":
    app = HandwritingRecognizer(model, canvas_size=280, pen_width=15)
    app.run()
