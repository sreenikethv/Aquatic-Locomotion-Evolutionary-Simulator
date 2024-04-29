import taichi as ti
import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import copy


real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

weights = scalar()
bias = scalar()
actuation = scalar()
actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
x, v = vec(), vec()
C, F = mat(), mat()
loss = scalar()
x_avg = vec()

n_sin_waves = 4
actuation_omega = 20
act_strength = 4

def allocate_fields(mutate=False):
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)
    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id)
    ti.root.dense(ti.i, n_particles).place(particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(v)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(C)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in)
    ti.root.dense(ti.ij, n_grid).place(grid_m_in)
    ti.root.dense(ti.ij, n_grid).place(grid_v_out)

    ti.root.place(loss, x_avg)

    ti.root.lazy_grad()

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]


@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)

        act_id = actuator_id[p]

        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        # ti.print(act)

        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base +
                          offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass


bound = 3
coeff = 0.5


@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out


@ti.kernel
def g2p(f: ti.i32): # graph 2 particle?
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            act += weights[i, j] * ti.sin(actuation_omega * t * dt + 2 * math.pi / n_sin_waves * j)
        act += bias[i]
        actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])


@ti.kernel
def compute_loss():
    dist = x_avg[None][0]
    loss[None] = -dist


@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)


@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        advance(s)
    x_avg[None] = [0, 0]
    compute_x_avg()
    compute_loss()


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def __str__(self):
        return f"{self.n_particles}, {self.n_solid_particles}"

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def clear_rects(self):
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        global n_particles

        self.n_particles = 0
        self.n_solid_particles = 0

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

class Morphology:
    def __init__(self, rectangles):
        self.rectangles = rectangles
    def get_num_rectangles(self):
        return len(self.rectangles)
    def get_rectangle(self, index):
        specific_rect = self.rectangles[index]
        assert len(specific_rect) == 6
        return specific_rect
    def get_rectangles(self):
        return self.rectangles
    def modify_rectangle(self, index, x, y, w, h, actuation, ptype):
        self.rectangles[index] = [x, y, w, h, actuation, ptype]
    def add_rectangle(self, x, y, w, h, actuation, ptype):
        self.rectangles.append([x, y, w, h, actuation, ptype])
    def remove_rectangle(self, index):
        del self.rectangles[index]

def generate_shape(rectangles):
    min_x = min(rect[0] for rect in rectangles)
    min_y = min(rect[1] for rect in rectangles)
    max_x = max(rect[0] + rect[2] for rect in rectangles)
    max_y = max(rect[1] + rect[3] for rect in rectangles)

    if (min_x < 0):
        min_x = 0
    if (min_y < 0):
        min_y = 0

    # grid_size = 0.05  # size of grid squares
    min_w = min(rect[2] for rect in rectangles)
    min_h = min(rect[3] for rect in rectangles)
    grid_size = min(min_w, min_h)

    num_cols = int((max_x - min_x) // grid_size)
    num_rows = int((max_y - min_y) // grid_size)
    grid = []
    for _ in range(num_rows):
        row = [False] * num_cols
        grid.append(row)

    # mark filled squares based on existing rectangles
    for rect in rectangles:
        x, y, w, h, p, act = rect
        col_start = int((x - min_x) // grid_size)
        col_end = int((x + w - min_x) // grid_size)
        row_start = int((y - min_y) // grid_size)
        row_end = int((y + h - min_y) // grid_size)
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                grid[row][col] = True 

    # identify empty and contiguous squares and select
    empty_contiguous_squares = []
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if not grid[row][col]:
                rect = [min_x + col * grid_size, min_y + row * grid_size, grid_size, grid_size, 0, 1]
                contiguous_found = False
                for existing_rect in rectangles:
                    if are_contiguous(rect, existing_rect):
                        contiguous_found = True
                        break
                if not contiguous_found:
                    empty_contiguous_squares.append(rect)

    # select an empty and contiguous square
    if empty_contiguous_squares:
        new_rect = random.choice(empty_contiguous_squares)
        return new_rect
    else:
        return None

def is_contiguous(rectangles):
    for i, rect1 in enumerate(rectangles):
        for j, rect2 in enumerate(rectangles):
            if i != j:
                if not are_contiguous(rect1, rect2):
                    return False
    return True

def are_contiguous(rect1, rect2):
    if are_overlapping(rect1, rect2):
        return False
    x1, y1, w1, h1, p1, act1 = rect1
    x2, y2, w2, h2, p2, act2 = rect2
    if (x1 == x2 + w2 or x2 == x1 + w1) and \
       (y1 == y2 + h2 or y2 == y1 + h1):
        return True  # rectangles share a border
    return False 

def are_overlapping(rect1, rect2):
    x1, y1, w1, h1, p1, act1 = rect1 
    x2, y2, w2, h2, p2, act2 = rect2
    if (x1 + w1 >= x2 and x1 + w1 <= x2 + w2):
        if (y1 + h1 >= y2 and y1 + h1 <= y2 + h2):
            return True
    elif (x2 + w2 >= x1 and x2 + w2 <= x1 + w1):
        if (y2 + h2 >= y1 and y2 + h2 <= y1 + h1):
            return True
    return False

def make_contiguous(rectangles, new_rect):
    # modify new_rect to make it contiguous with the overlapping rectangle
    for rect in rectangles:
            if not are_contiguous(rect, new_rect):
                if rect[1] > new_rect[1]:
                    new_rect[3] = rect[1] - new_rect[1]
                elif rect[1] < new_rect[1]:
                    old_y = new_rect[1]
                    new_rect[1] = rect[1] + rect[3]
                    new_rect[3] = old_y + new_rect[3] - new_rect[1]
    return new_rect

    
def mutate_morphology(scene, morphology):
    has_mutated = False
    tryCount = 0
    while (has_mutated == False and tryCount <= 100):
        tryCount+=1
        if tryCount%10==0:
            print("TRY ", tryCount)
        particle_count_changed = True
        original_rectangles = copy.deepcopy(morphology.rectangles)
        new_rectangles = copy.deepcopy(morphology.rectangles)
        index = random.randint(0, morphology.get_num_rectangles() - 1)
        operation = random.choice(['resize_shape', 'move_shape', 'add_shape', 'remove_shape'])
        #operation = 'add_shape'

        if operation == 'move_shape':
            displacement=random.uniform(-0.05, 0.05)
            direction = random.choice(['horizontal', 'vertical'])
            if (direction == 'horizontal'):
                #new_x = max(min(x + displacement, 1.0 - w), 0.0)  # Ensure it doesn't exceed scene boundaries
                new_rectangles[index][0] = round(new_rectangles[index][0]+displacement, 2)
            else:
                new_rectangles[index][1] = round(new_rectangles[index][1]+displacement, 2)
            particle_count_changed = False
        
        elif operation == 'resize_shape':
            direction = random.choice(['increase', 'decrease'])
            x, y, w, h, actuation, ptype = new_rectangles[index]
            resize_amount = random.uniform(0.01, 0.05)

            if random.choice(['width', 'height']) == 'width':
                new_width = w if direction == 'decrease' else min(w + resize_amount, 1.0 - x)
            else:
                new_height = h if direction == 'decrease' else min(h + resize_amount, 1.0 - y)

        elif operation == 'add_shape':
            new_rectangles.append(generate_shape(new_rectangles))
        
        elif operation == 'remove_shape':
            if (new_rectangles[index][4] == 0): #actuation is 0
                del new_rectangles[index]

        has_mutated = compare_lists(original_rectangles, new_rectangles)
        if not is_contiguous(new_rectangles):
            if (operation == 'add_shape'):
                #new_rect = make_contiguous(new_rectangles, new_rectangles[len(new_rectangles)-1])
                #new_rectangles = copy.deepcopy(morphology.rectangles).append(new_rect)
                print("MAKE CONTIGUOUS")
            else:
                has_mutated = False

        if has_mutated == False:
            print("DIFFERENCE NOT FOUND")

    return new_rectangles, particle_count_changed

def compare_lists(list1, list2):
    if len(list1) != len(list2):
        return True

    for sub1, sub2 in zip(list1, list2):
        if len(sub1) != len(sub2):
            return True
        if sub1 != sub2:
            return True

    return False


def fish(scene, morphology, calculate_particles=True, mutate=False): 
    # if (mutate == True):
    #     scene.clear_rects()
    actuators_count = 0
    for rectangle in morphology.rectangles:
        #print("FISH:", rectangle)
        x, y, w, h, actuation, ptype = rectangle
        scene.add_rect(x, y, w, h, actuation, ptype)
        if actuation > -1:
            actuators_count += 1
    scene.set_n_actuators(actuators_count)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

def visualize(s, folder):
    aid = actuator_id.to_numpy()
    colors = np.empty(shape=n_particles, dtype=np.uint32)
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            act = actuation_[s - 1, int(aid[i])]
            color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
        colors[i] = color
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)

    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')

def initialize_weights(scale=0.01):
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            weights[i, j] = np.random.randn() * scale
            #weights2[i, j] = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()

    # initialization
    scene = Scene()

    #walker
    scene.set_offset(0.1, 0.03)
    initial_rectangles = [
        [0.0, 0.1, 0.3, 0.1, -1, 1],
        [0.0, 0.0, 0.05, 0.1, 0, 1],
        [0.05, 0.0, 0.05, 0.1, 1, 1],
        [0.2, 0.0, 0.05, 0.1, 2, 1],
        [0.25, 0.0, 0.05, 0.1, 3, 1],
    ]

    # initial_rectangles = [
    #                     [0.025, 0.025, 0.95, 0.1, -1, 0],
    #                     [0.2, 0.2, 0.15, 0.05, -1, 1],
    #                     [0.15, 0.15, 0.03, 0.1, 0, 1],
    # ]

    # scene.set_offset(0.1, 0.03)
    # initial_rectangles = [[0.025, 0.025, 0.95, 0.1, -1, 0],
    #                     [0.0, 0.05, 0.15, 0.1, -1, 1],
    #                     [0.05, 0.0, 0.05, 0.05, 0, 1]
    # ]


    # initial_rectangles = [
    #                     [0.025, 0.025, 0.95, 0.1, -1, 0],
    #                     [0.1, 0.2, 0.15, 0.05, -1, 1],
    #                     [0.1, 0.15, 0.03, 0.05, 0, 1],
    #                     [0.13, 0.15, 0.03, 0.03, 0, 1]
    # ]

    # initial_rectangles = [
    #                     [0.025, 0.025, 0.95, 0.1, -1, 0],
    #                     [0.1, 0.2, 0.15, 0.05, -1, 1],
    #                     [0.1, 0.15, 0.03, 0.05, 0, 1],
    #                     # [0.13, 0.15, 0.03, 0.03, 0, 1]
    #                 ]

    #one-flipper
    # scene.set_offset(0.1, 0.03)
    # initial_rectangles = [
    #     [0.025, 0.025, 0.95, 0.1, -1, 0],
    #     [0.15, 0.25, 0.4, 0.15, -1, 1],
    #     [0.15, 0.15, 0.05, 0.1, 0, 1],
    # ]

    # initial_rectangles = [
    #     [0.025, 0.025, 0.95, 0.1, -1, 0],
    #     [0.1, 0.1, 0.2, 0.05, -1, 1],
    #     [0.1, 0.15, 0.05, 0.1, 0, 1],
    # ]

    initial_morphology = Morphology(initial_rectangles)
    fish(scene, initial_morphology)
    current_morphology = initial_morphology
    scene.finalize()
    allocate_fields()
    initialize_weights()
    for i in range(scene.n_particles):
        x[0, i] = scene.x[i]
        F[0, i] = [[1, 0], [0, 1]]
        actuator_id[i] = scene.actuator_id[i]
        particle_type[i] = scene.particle_type[i]

    MUTATE_ROUND = 20
    NUM_DESCENDANTS = 3
    best_loss = float('inf')
    best_mutated_morphology = None
    trycount = 0
    losses = []

    for iter in range(options.iters):
        if iter % MUTATE_ROUND == 0 and iter > 0:         
            best_loss_descendant = float('inf')

            # generate and evaluate multiple mutations
            for _ in range(NUM_DESCENDANTS):
                trycount += 1
                print("trydescendant",trycount)
                new_rectangles, particle_count_changed = mutate_morphology(scene, current_morphology)
                new_morphology = Morphology(new_rectangles)
                fish(scene, new_morphology, mutate=True)

                # trains weights and biases for each child generation
                initialize_weights()
                for iter in range(20):
                    with ti.ad.Tape(loss):
                        forward()
                    l = loss[None]
                    for i in range(n_actuators):
                        for j in range(n_sin_waves):
                            weights[i, j] -= learning_rate * weights.grad[i, j]
                        bias[i] -= learning_rate * bias.grad[i]
                
                if l < best_loss_descendant:
                    best_loss_descendant = l
                    best_mutated_morphology = Morphology(new_morphology.rectangles)   
            
            print('i=', iter, 'best_loss_descendant=', best_loss_descendant)
            losses.append(best_loss_descendant)

            # update best morphology
            if best_loss_descendant < best_loss:
                best_loss = best_loss_descendant
                current_morphology = Morphology(best_mutated_morphology.rectangles)
            else:
                best_mutated_morphology = Morphology(current_morphology.rectangles)

            for r in current_morphology.rectangles:
                print(r) #"FISH:"

        with ti.ad.Tape(loss):
            forward()
        l = loss[None]
            
        losses.append(l)
        best_loss = l
        print('i=', iter, 'loss=', l)
        learning_rate = 0.1
    
        for i in range(n_actuators):
            for j in range(n_sin_waves):
                weights[i, j] -= learning_rate * weights.grad[i, j]
            bias[i] -= learning_rate * bias.grad[i]

        if iter % 10 == 0:
            # visualize
            forward(1500)
            for s in range(15, 1500, 16):
                visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

    # ti.profiler_print()
    plt.title("Optimization of Initial Velocity")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    main()
