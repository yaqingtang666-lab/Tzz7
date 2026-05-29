import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长 (半隐式稳定上限约 0.01，当前远低于此)
k_s = 10000.0      # 结构弹簧劲度系数，控制拉伸与压缩
k_sh = 5000.0      # 剪切弹簧劲度系数，控制正方形变平行四边形程度
k_b = 1000.0       # 弯曲弹簧劲度系数,控制折叠与弯曲程度
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸

# 球体碰撞参数
sphere_center = ti.Vector.field(3, dtype=float, shape=())
sphere_radius = ti.field(dtype=float, shape=())
sphere_center[None] = ti.Vector([0.0, -0.2, 0.0])
sphere_radius[None] = 0.3

# 球体渲染粒子参数
SPHERE_RES = 15
num_sphere_points = SPHERE_RES * SPHERE_RES
sphere_points = ti.Vector.field(3, dtype=float, shape=num_sphere_points)

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)  # 隐式欧拉专用力场

# 弹簧数据场
max_springs = N * N * 8  # 增加弹簧数量上限
spring_indices = ti.field(dtype=int, shape=max_springs * 2) # 用于渲染画线
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_stiffness = ti.field(dtype=float, shape=max_springs)  # 每个弹簧的劲度系数
num_springs = ti.field(dtype=int, shape=())

def init_sphere_points():
    """初始化球体渲染点"""
    idx = 0
    for i in range(SPHERE_RES):
        for j in range(SPHERE_RES):
            theta = 2 * math.pi * i / SPHERE_RES
            phi = math.pi * (j + 0.5) / SPHERE_RES
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            sphere_points[idx] = ti.Vector([x, y, z])
            idx += 1

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化弹簧 (结构弹簧 + 剪切弹簧 + 弯曲弹簧)"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        if i < N - 1:
            idx_right = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_right])
            spring_lengths[c] = (x[idx] - x[idx_right]).norm()
            spring_stiffness[c] = k_s
        if j < N - 1:
            idx_down = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_down])
            spring_lengths[c] = (x[idx] - x[idx_down]).norm()
            spring_stiffness[c] = k_s
        
        if i < N - 1 and j < N - 1:
            idx_diag1 = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag1])
            spring_lengths[c] = (x[idx] - x[idx_diag1]).norm()
            spring_stiffness[c] = k_sh
        if i < N - 1 and j > 0:
            idx_diag2 = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_diag2])
            spring_lengths[c] = (x[idx] - x[idx_diag2]).norm()
            spring_stiffness[c] = k_sh
        
        if i < N - 2:
            idx_skip_i = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip_i])
            spring_lengths[c] = (x[idx] - x[idx_skip_i]).norm()
            spring_stiffness[c] = k_b
        if j < N - 2:
            idx_skip_j = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, idx_skip_j])
            spring_lengths[c] = (x[idx] - x[idx_skip_j]).norm()
            spring_stiffness[c] = k_b

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    """从 Python 层按顺序调用各初始化 kernel，确保 GPU 同步"""
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 弹簧力)"""
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            k = spring_stiffness[i]
            f_spring = -k * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def handle_sphere_collision(pos: ti.template(), vel: ti.template()):
    """处理球体碰撞"""
    center = sphere_center[None]
    radius = sphere_radius[None]
    for i in range(N * N):
        if is_fixed[i] == 0:
            d = pos[i] - center
            dist = d.norm()
            if dist < radius:
                normal = d / dist
                pos[i] = center + normal * radius
                vel[i] = vel[i] - vel[i].dot(normal) * normal * 0.8

@ti.kernel
def step_explicit():
    """显式欧拉"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
    handle_sphere_collision(x, v)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
    handle_sphere_collision(x, v)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
        handle_sphere_collision(x_next, v_next)
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

@ti.kernel
def update_sphere_vertices(sphere_verts: ti.template()):
    """更新球体顶点位置"""
    center = sphere_center[None]
    radius = sphere_radius[None]
    for i in range(num_sphere_points):
        sphere_verts[i] = center + sphere_points[i] * radius

def main():
    init_cloth()
    init_sphere_points()

    window = ti.ui.Window("Mass Spring System with Shear & Bending Springs", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    sphere_verts = ti.Vector.field(3, dtype=float, shape=num_sphere_points)

    current_method = 1
    paused = False

    while window.running:
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.38, 0.4)

        window.GUI.text("Integration Method:")

        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.text("")
        window.GUI.text("Spring Stiffness:")
        global k_s, k_sh, k_b
        k_s = window.GUI.slider_float("Structural", k_s, 1000, 20000)
        k_sh = window.GUI.slider_float("Shear", k_sh, 1000, 10000)
        k_b = window.GUI.slider_float("Bending", k_b, 100, 5000)

        window.GUI.text("")
        window.GUI.text("Sphere Settings:")
        sphere_center[None][0] = window.GUI.slider_float("Sphere X", sphere_center[None][0], -1.0, 1.0)
        sphere_center[None][1] = window.GUI.slider_float("Sphere Y", sphere_center[None][1], -1.0, 1.0)
        sphere_center[None][2] = window.GUI.slider_float("Sphere Z", sphere_center[None][2], -1.0, 1.0)
        sphere_radius[None] = window.GUI.slider_float("Sphere Radius", sphere_radius[None], 0.1, 0.5)

        window.GUI.end()

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        update_sphere_vertices(sphere_verts)

        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        scene.particles(sphere_verts, radius=sphere_radius[None] * 0.12, color=(0.8, 0.4, 0.4))

        canvas.scene(scene)

        window.show()

if __name__ == '__main__':
    main()