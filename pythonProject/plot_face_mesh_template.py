import os
import urllib.request
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# 設定中文字型
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Google MediaPipe 官方標準 468 點人臉 OBJ
CANONICAL_OBJ_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_geometry/data/canonical_face_model.obj"
)
LOCAL_OBJ_FILE = "canonical_face_model.obj"

# ==============================================================================
# 特徵點分類資料與連線定義
# ==============================================================================

VERTEX_GROUPS = {
    "Rigid Points (剛性基準點)": {
        "color": "#1f77b4",  # 寶藍色
        "marker": "s",
        "size": 65,
        "points": {
            1: "鼻尖",
            6: "鼻樑中央",
            197: "鼻樑上段",
            195: "骨骼連線",
            168: "印堂",
            33: "右眼內眥",
            263: "左眼內眥",
            152: "下巴下緣",
        },
    },
    "Brows & Forehead (眉部與額頭)": {
        "color": "#ff7f0e",  # 橘色
        "marker": "^",
        "size": 65,
        "points": {
            70: "右皺眉肌***",
            63: "右眉中",
            105: "右眉峰",
            300: "左皺眉肌***",
            293: "左眉中",
            334: "左眉峰",
            9: "額頭中央",
        },
    },
    "Eyes & Crow's Feet (眼瞼與眼角)": {
        "color": "#2ca02c",  # 翠綠色
        "marker": "o",
        "size": 60,
        "points": {
            159: "右眼上瞼",
            145: "右眼下瞼",
            386: "左眼上瞼",
            374: "左眼下瞼",
            130: "右外眼角(真誠微笑)***",
            246: "右眼外周緣",
            359: "左外眼角(真誠微笑)***",
            466: "左眼外周緣",
        },
    },
    "Mouth & Speech (唇部與說話/微笑)": {
        "color": "#d62728",  # 鮮紅色
        "marker": "D",
        "size": 60,
        "points": {
            0: "上唇正中(說話)***",
            17: "下唇正中(說話)***",
            61: "右嘴角(微笑)***",
            291: "左嘴角(微笑)***",
            13: "上唇內咬合緣",
            14: "下唇內咬合緣",
        },
    },
    "Jaw & Cheeks (臉頰與下頜咬合肌)": {
        "color": "#9467bd",  # 紫色
        "marker": "P",
        "size": 75,
        "points": {
            234: "右臉頰最外側",
            454: "左臉頰最外側",
            172: "右下頜角(咬合肌)***",
            397: "左下頜角(咬合肌)***",
        },
    },
}

STARRED_VERTICES = {70, 300, 61, 291, 0, 17, 130, 359, 172, 397}

FUNCTIONAL_PAIRS = [
    (70, 300, "皺眉肌連線 (70-300)", "#ff7f0e"),
    (130, 359, "真誠微笑眼角 (130-359)", "#2ca02c"),
    (61, 291, "微笑嘴角連線 (61-291)", "#d62728"),
    (0, 17, "說話雙唇開合 (0-17)", "#d62728"),
    (172, 397, "下頜咬合肌 (172-397)", "#9467bd"),
]

MIDLINE_IDS = [9, 168, 6, 197, 195, 1, 0, 17, 152]

CUSTOM_LABEL_OFFSETS = {
    168: (18, 6), 6: (-22, 4), 197: (22, 0), 195: (-24, -2), 1: (20, -6),
    9: (0, 14), 70: (-22, 10), 63: (-12, 14), 105: (10, 14),
    300: (22, 10), 293: (12, 14), 334: (-10, 14),
    33: (14, -10), 159: (0, 12), 145: (0, -14), 130: (-28, -8), 246: (-24, 10),
    263: (-14, -10), 386: (0, 12), 374: (0, -14), 359: (28, -8), 466: (24, 10),
    0: (-28, 14), 13: (28, 12), 14: (28, -12), 17: (-28, -16), 61: (-26, 0), 291: (26, 0),
    234: (-26, 0), 454: (26, 0), 172: (-28, -8), 397: (28, -8), 152: (0, -16),
}


# ==============================================================================
# 模型載入工具
# ==============================================================================

def load_canonical_face_mesh():
    if not os.path.exists(LOCAL_OBJ_FILE):
        urllib.request.urlretrieve(CANONICAL_OBJ_URL, LOCAL_OBJ_FILE)

    vertices, faces = [], []
    with open(LOCAL_OBJ_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                face_indices = [int(p.split("/")[0]) - 1 for p in parts]
                if len(face_indices) == 3:
                    faces.append(face_indices)
                elif len(face_indices) == 4:
                    faces.append([face_indices[0], face_indices[1], face_indices[2]])
                    faces.append([face_indices[0], face_indices[2], face_indices[3]])

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)


# ==============================================================================
# 互動式主應用程式 (Tkinter + Matplotlib)
# ==============================================================================

class FaceMeshInteractiveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe 468 Face Mesh 特徵點互動檢視器")
        self.root.geometry("1480x920")
        self.root.minsize(1150, 750)

        # 載入模型數據
        try:
            self.vertices, self.faces = load_canonical_face_mesh()
        except Exception as e:
            messagebox.showerror("模型下載失敗", f"無法下載或載入 canonical_face_model.obj:\n{e}")
            raise e

        self.x = self.vertices[:, 0]
        self.y = self.vertices[:, 1]
        self.triang = mtri.Triangulation(self.x, self.y, self.faces)

        # 構建頂點反向查詢字典
        self.vid_to_meta = {}
        for g_name, g_data in VERTEX_GROUPS.items():
            for vid, desc in g_data["points"].items():
                self.vid_to_meta[vid] = {
                    "group": g_name,
                    "desc": desc,
                    "color": g_data["color"],
                    "marker": g_data["marker"],
                    "size": g_data["size"],
                    "starred": vid in STARRED_VERTICES,
                }

        # 互動狀態變數
        self.custom_selected_vids = set()
        self.group_vars = {}
        self.show_mesh_var = tk.BooleanVar(value=True)
        self.show_all_pts_var = tk.BooleanVar(value=True)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_functional_var = tk.BooleanVar(value=True)
        self.show_midline_var = tk.BooleanVar(value=True)
        self.only_starred_var = tk.BooleanVar(value=False)

        # 配置風格與文字大小
        self._configure_styles()

        # 建立 UI 與初次繪製
        self._setup_ui()
        self.redraw_mesh()

    def _configure_styles(self):
        """配置整體 Panel 字型大小與外觀樣式"""
        style = ttk.Style()
        style.configure(".", font=("Microsoft JhengHei", 10))
        style.configure("TLabelframe", padding=10)
        style.configure("TLabelframe.Label", font=("Microsoft JhengHei", 11, "bold"), foreground="#1a1a1a")
        style.configure("TCheckbutton", font=("Microsoft JhengHei", 10))
        style.configure("TButton", font=("Microsoft JhengHei", 10, "bold"), padding=4)
        style.configure("TLabel", font=("Microsoft JhengHei", 10))

    def _setup_ui(self):
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        # 左側滾動控制面板 (加寬至 460px 容納較大文字)
        sidebar_frame = ttk.Frame(main_paned, width=460, padding=10)
        main_paned.add(sidebar_frame, weight=0)

        # 右側繪圖區
        canvas_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(canvas_frame, weight=1)

        self._build_sidebar(sidebar_frame)
        self._build_canvas(canvas_frame)

    def _build_sidebar(self, parent):
        canvas_scroll = tk.Canvas(parent, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas_scroll.yview)
        scroll_content = ttk.Frame(canvas_scroll)

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=scroll_content, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ---------------- Section 1: 臉部部位篩選 ----------------
        group_box = ttk.LabelFrame(scroll_content, text="1. 臉部部位高亮選取", padding=10)
        group_box.pack(fill=tk.X, pady=(0, 10))

        for g_name, g_data in VERTEX_GROUPS.items():
            var = tk.BooleanVar(value=True)
            self.group_vars[g_name] = var
            cb = ttk.Checkbutton(
                group_box,
                text=f"{g_name} ({len(g_data['points'])} 點)",
                variable=var,
                command=self.redraw_mesh
            )
            cb.pack(anchor="w", pady=3)

        starred_cb = ttk.Checkbutton(
            group_box,
            text="★ 僅篩選核心非語言溝通點 (*** Starred Only)",
            variable=self.only_starred_var,
            command=self.redraw_mesh
        )
        starred_cb.pack(anchor="w", pady=(8, 4))

        btn_row_group = ttk.Frame(group_box)
        btn_row_group.pack(fill=tk.X, pady=(6, 2))
        ttk.Button(btn_row_group, text="全部勾選", command=lambda: self._set_all_groups(True)).pack(side=tk.LEFT,
                                                                                                    expand=True,
                                                                                                    fill=tk.X, padx=3)
        ttk.Button(btn_row_group, text="全部取消", command=lambda: self._set_all_groups(False)).pack(side=tk.LEFT,
                                                                                                     expand=True,
                                                                                                     fill=tk.X, padx=3)

        # ---------------- Section 2: 指定特徵點 ID 高亮 ----------------
        point_box = ttk.LabelFrame(scroll_content, text="2. 指定 Vertex ID 搜尋 / 高亮", padding=10)
        point_box.pack(fill=tk.X, pady=10)

        entry_label = ttk.Label(point_box, text="輸入頂點編號 (可多個，如 1, 61, 291):", font=("Microsoft JhengHei", 10))
        entry_label.pack(anchor="w", pady=(0, 3))

        self.vid_entry_var = tk.StringVar()
        entry_row = ttk.Frame(point_box)
        entry_row.pack(fill=tk.X, pady=4)

        entry = tk.Entry(
            entry_row,
            textvariable=self.vid_entry_var,
            font=("Microsoft JhengHei", 11),
            relief="solid",
            bd=1
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=3)
        entry.bind("<Return>", lambda e: self._on_apply_entry_vids())

        ttk.Button(entry_row, text="加入高亮", command=self._on_apply_entry_vids).pack(side=tk.RIGHT)

        ttk.Label(point_box, text="預定義關鍵點快速點選清單:", font=("Microsoft JhengHei", 10)).pack(anchor="w",
                                                                                                     pady=(8, 3))
        list_frame = ttk.Frame(point_box)
        list_frame.pack(fill=tk.X)

        self.preset_listbox = tk.Listbox(
            list_frame,
            height=6,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            font=("Microsoft JhengHei", 10),
            activestyle="none"
        )
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.preset_listbox.yview)
        self.preset_listbox.configure(yscrollcommand=list_scroll.set)

        self.preset_vids = []
        for g_data in VERTEX_GROUPS.values():
            for vid, desc in g_data["points"].items():
                is_s = "★ " if vid in STARRED_VERTICES else "   "
                self.preset_listbox.insert(tk.END, f"#{vid:>3} : {is_s}{desc}")
                self.preset_vids.append(vid)

        self.preset_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.preset_listbox.bind("<<ListboxSelect>>", self._on_preset_list_select)

        ttk.Button(point_box, text="清除自訂選取點位", command=self._clear_custom_selection).pack(fill=tk.X,
                                                                                                  pady=(8, 2))

        # ---------------- Section 3: 顯示輔助設定 ----------------
        display_box = ttk.LabelFrame(scroll_content, text="3. 畫面顯示設定", padding=10)
        display_box.pack(fill=tk.X, pady=10)

        ttk.Checkbutton(display_box, text="顯示三角網格線 (Wireframe 加粗)", variable=self.show_mesh_var,
                        command=self.redraw_mesh).pack(anchor="w", pady=3)
        ttk.Checkbutton(display_box, text="顯示全部 468 點背景點", variable=self.show_all_pts_var,
                        command=self.redraw_mesh).pack(anchor="w", pady=3)
        ttk.Checkbutton(display_box, text="顯示頂點編號與中文標籤", variable=self.show_labels_var,
                        command=self.redraw_mesh).pack(anchor="w", pady=3)
        ttk.Checkbutton(display_box, text="顯示功能連動虛線 (*** Pairs)", variable=self.show_functional_var,
                        command=self.redraw_mesh).pack(anchor="w", pady=3)
        ttk.Checkbutton(display_box, text="顯示剛性中軸線 (Midline)", variable=self.show_midline_var,
                        command=self.redraw_mesh).pack(anchor="w", pady=3)

        # ---------------- Section 4: 點選資訊卡 ----------------
        info_box = ttk.LabelFrame(scroll_content, text="4. 點選特徵點資訊 (可在圖上直接點選)", padding=10)
        info_box.pack(fill=tk.X, pady=10)

        self.info_text_var = tk.StringVar(value="提示：可在右側人臉圖上直接點選任意點，程式將自動吸附並顯示資訊。")
        info_label = ttk.Label(
            info_box,
            textvariable=self.info_text_var,
            wraplength=410,
            justify="left",
            font=("Microsoft JhengHei", 10)
        )
        info_label.pack(fill=tk.X, pady=2)

    def _build_canvas(self, parent):
        self.fig, self.ax = plt.subplots(figsize=(8.5, 9.5), dpi=100)
        self.fig.patch.set_facecolor("#ffffff")
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, parent)
        self.toolbar.update()

        self.fig.canvas.mpl_connect("button_press_event", self._on_canvas_click)

    # ==========================================================================
    # 邏輯控制與選取處理
    # ==========================================================================

    def _set_all_groups(self, state: bool):
        for var in self.group_vars.values():
            var.set(state)
        self.redraw_mesh()

    def _on_apply_entry_vids(self):
        text = self.vid_entry_var.get().strip()
        if not text:
            return
        parts = text.replace("，", ",").split(",")
        valid_added = 0
        for p in parts:
            p = p.strip()
            if p.isdigit():
                vid = int(p)
                if 0 <= vid < len(self.vertices):
                    self.custom_selected_vids.add(vid)
                    valid_added += 1
        if valid_added == 0:
            messagebox.showwarning("格式錯誤", "請輸入 0 ~ 467 之間的數字編號！")
        else:
            self.redraw_mesh()

    def _on_preset_list_select(self, event):
        indices = self.preset_listbox.curselection()
        for idx in indices:
            self.custom_selected_vids.add(self.preset_vids[idx])
        self.redraw_mesh()

    def _clear_custom_selection(self):
        self.custom_selected_vids.clear()
        self.preset_listbox.selection_clear(0, tk.END)
        self.redraw_mesh()

    def _on_canvas_click(self, event):
        if self.toolbar.mode != "":
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = self.x - event.xdata
        dy = self.y - event.ydata
        dists = np.hypot(dx, dy)
        nearest_idx = int(np.argmin(dists))

        if dists[nearest_idx] < 8.0:
            if nearest_idx in self.custom_selected_vids:
                self.custom_selected_vids.remove(nearest_idx)
            else:
                self.custom_selected_vids.add(nearest_idx)

            meta = self.vid_to_meta.get(nearest_idx)
            coords = self.vertices[nearest_idx]
            info = f"【選中頂點 #{nearest_idx}】\n"
            if meta:
                info += f"● 名稱：{meta['desc']}\n● 部位：{meta['group']}\n"
                if meta["starred"]:
                    info += "★ 核心非語言溝通特徵點\n"
            else:
                info += "● 名稱：一般網格頂點 (未定義自訂標籤)\n"
            info += f"● 3D 空間座標 (X, Y, Z)：\n  [{coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}]"
            self.info_text_var.set(info)

            self.redraw_mesh()

    # ==========================================================================
    # 核心繪圖渲染 (加粗線條)
    # ==========================================================================

    def redraw_mesh(self):
        self.ax.clear()

        # 1. 繪製三角網格背景線 (加粗至 linewidth=1.2)
        if self.show_mesh_var.get():
            self.ax.triplot(
                self.triang,
                color="#b5b5b5",
                linewidth=1.2,
                alpha=0.85,
                zorder=1
            )

        # 2. 繪製所有 468 點背景點 (點徑微調至 s=9)
        if self.show_all_pts_var.get():
            self.ax.scatter(self.x, self.y, color="#999999", s=9, alpha=0.55, zorder=2)

        # 3. 繪製剛性中軸線 (加粗至 linewidth=2.4)
        if self.show_midline_var.get():
            self.ax.plot(
                self.x[MIDLINE_IDS], self.y[MIDLINE_IDS],
                color="#1f77b4", linestyle=":", linewidth=2.4, alpha=0.75, zorder=3,
                label="人臉中軸線"
            )

        # 4. 繪製功能對稱連線 (加粗至 linewidth=2.5)
        if self.show_functional_var.get():
            for v1, v2, _, p_color in FUNCTIONAL_PAIRS:
                self.ax.plot(
                    [self.x[v1], self.x[v2]], [self.y[v1], self.y[v2]],
                    color=p_color, linestyle="--", linewidth=2.5, alpha=0.88, zorder=4
                )

        # 5. 計算當前應高亮的頂點集合
        highlight_dict = {}
        only_starred = self.only_starred_var.get()

        for g_name, is_checked in self.group_vars.items():
            if is_checked.get():
                g_data = VERTEX_GROUPS[g_name]
                for vid, desc in g_data["points"].items():
                    if only_starred and vid not in STARRED_VERTICES:
                        continue
                    highlight_dict[vid] = {
                        "color": g_data["color"],
                        "marker": g_data["marker"],
                        "size": g_data["size"],
                        "desc": desc,
                        "starred": vid in STARRED_VERTICES,
                    }

        # 加入自訂選取頂點
        for vid in self.custom_selected_vids:
            if vid in self.vid_to_meta:
                meta = self.vid_to_meta[vid]
                highlight_dict[vid] = {
                    "color": meta["color"],
                    "marker": meta["marker"],
                    "size": meta["size"] * 1.35,
                    "desc": meta["desc"],
                    "starred": meta["starred"],
                }
            else:
                highlight_dict[vid] = {
                    "color": "#e377c2",
                    "marker": "*",
                    "size": 110,
                    "desc": "自訂選取",
                    "starred": False,
                }

        # 6. 渲染所有高亮點與文字標籤
        show_labels = self.show_labels_var.get()

        for vid, prop in highlight_dict.items():
            vx, vy = self.x[vid], self.y[vid]
            is_starred = prop["starred"]

            # 散點圖標記
            self.ax.scatter(
                vx, vy,
                c=prop["color"],
                marker=prop["marker"],
                s=prop["size"],
                edgecolors="black",
                linewidths=1.0,
                zorder=6,
            )

            # 核心點 / 選取點的外外圈加粗強調
            if is_starred or vid in self.custom_selected_vids:
                self.ax.scatter(
                    vx, vy,
                    facecolors="none",
                    edgecolors="#ff1493" if vid in self.custom_selected_vids else "black",
                    s=prop["size"] * 2.3,
                    linewidths=1.8,
                    linestyle="--",
                    zorder=7,
                )

            # 標籤文字 (稍微增大字級至 9.5)
            if show_labels:
                dx, dy = CUSTOM_LABEL_OFFSETS.get(vid, (0, 10))
                tag_str = f"#{vid} {prop['desc']}" if prop['desc'] else f"#{vid}"
                self.ax.annotate(
                    tag_str,
                    xy=(vx, vy),
                    xytext=(dx * 1.35, dy * 1.25),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold" if is_starred else "normal",
                    color="#111111",
                    bbox=dict(
                        boxstyle="round,pad=0.22",
                        fc="#fff9e6" if is_starred else "white",
                        ec=prop["color"],
                        lw=1.2,
                        alpha=0.94,
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=prop["color"],
                        lw=1.0,
                    ),
                    zorder=10,
                )

        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_title(
            f"MediaPipe 468 人臉網格 (目前高亮點數: {len(highlight_dict)})",
            fontsize=13,
            fontweight="bold",
            pad=10
        )
        self.canvas.draw_idle()


# ==============================================================================
# 程式入口
# ==============================================================================

if __name__ == "__main__":
    app_root = tk.Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    app = FaceMeshInteractiveApp(app_root)
    app_root.mainloop()