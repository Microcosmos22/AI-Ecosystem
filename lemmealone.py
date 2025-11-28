import pygame, random, math, numpy as np
pygame.init()
PRED_SIGHT_RADIUS = 200  # pixels

screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
W, H = screen.get_size()
clock = pygame.time.Clock()

def neighbour_vectors(agent, group, k=3, none = True):
    others = [a for a in group if a is not agent]
    nearest = sorted(others, key=lambda o: (o.x-agent.x)**2 + (o.y-agent.y)**2)[:k]

    vectors = []
    for n in nearest:
        vectors.extend([(n.x - agent.x) / agent.W, (n.y - agent.y) / agent.H])

    # Pad with zeros if fewer than k neighbours
    while len(vectors) < 2*k:
        vectors.extend([0.0, 0.0])
    if none:
        return np.zeros((2,2))

    return np.array(vectors)


def push_from_border(agent, margin=50, force=9.0):
    dx = 0.0
    dy = 0.0

    print(f"{agent.x < margin}, {agent.x > agent.W - margin}, {agent.y < margin}, {agent.y > agent.H}")
    print(dx, dy)
    if agent.x < margin:
        dx += force
    if agent.x > agent.W - margin:
        dx -= force
    if agent.y < margin:
        dy += force
    if agent.y > agent.H - margin:
        dy -= force
    return float(dx), float(dy)  # <-- return plain floats


def nearest_screen_edge(agent):
    """
    Returns the (x, y) coordinate of the nearest screen edge to the agent.
    """
    distances = {
        'left': agent.x,
        'right': agent.W - agent.x,
        'top': agent.y,
        'bottom': agent.H - agent.y
    }
    # find the nearest edge
    nearest_edge = min(distances, key=distances.get)

    if nearest_edge == 'left':
        return 0, agent.y
    elif nearest_edge == 'right':
        return agent.W, agent.y
    elif nearest_edge == 'top':
        return agent.x, 0
    else:  # bottom
        return agent.x, agent.H


def neighbour_offset(agent, group, k=3):
    others = [a for a in group if a is not agent]
    if not others:
        return 0.0, 0.0
    # sort by distance
    nearest = sorted(others, key=lambda o:(o.x-agent.x)**2+(o.y-agent.y)**2)[:k]
    dx = sum(o.x - agent.x for o in nearest) / max(1,len(nearest))
    dy = sum(o.y - agent.y for o in nearest) / max(1,len(nearest))
    return dx, dy

def normalize(v):
    d = math.hypot(v[0], v[1]) + 1e-6
    return np.array([v[0]/d, v[1]/d])

def mlp_fwd(x, w1, b1, w2, b2):
    h = np.tanh(x @ w1 + b1)
    o = np.tanh(h @ w2 + b2)
    return h, o



def mlp_back(x, h, o, target, w1, b1, w2, b2, lr):
    go = (o - target) * (1 - o**2)
    gw2 = np.outer(h, go)
    gb2 = go
    gh  = (go @ w2.T) * (1 - h**2)
    gw1 = np.outer(x, gh)
    gb1 = gh
    w1 -= lr*gw1; b1 -= lr*gb1
    w2 -= lr*gw2; b2 -= lr*gb2
    return w1,b1,w2,b2

class Agent:
    def __init__(self, color, kind, W, H, k=3):
        self.kind = kind
        self.color = color
        self.W = W
        self.H = H
        self.k = k  # number of neighbours to track
        self.respawn()

        if kind == "prey":
            in_size = 2 + 2 * k  # dx/dy to predator + dx/dy for each of k neighbours
        else:
            in_size = 4  # dx, dy + size, speed_factor

        self.w1 = np.zeros((in_size, 6))
        self.b1 = np.zeros(6)
        self.w2 = np.zeros((6, 2))
        self.b2 = np.zeros(2)
        self.lr = 0.02
        self.score = 0
        self.explore = 1.0

        self.x, self.y = 0, 0


    def respawn(self):
        self.x = random.random()*self.W
        self.y = random.random()*self.H
        self.vx = self.vy = 0
        self.size = 8
        self.speed_factor = 1.0

    def forward_and_move(self, target, neigh_offset=None):
        # compute NN input
        dx = (target[0] - self.x) / self.W
        dy = (target[1] - self.y) / self.H

        if self.kind == "prey" and neigh_offset is not None:
            ndx = neigh_offset[0] / self.W
            ndy = neigh_offset[1] / self.H
            inp = np.array([dx, dy, ndx, ndy])
        elif self.kind == "prey":
            inp = np.array([dx, dy, 0.0, 0.0])
        else:
            inp = np.array([dx, dy, self.size/40, self.speed_factor])

        h, out = mlp_fwd(inp, self.w1, self.b1, self.w2, self.b2)

        # compute velocity safely
        speed = 6 * (self.speed_factor if self.kind=="pred" else 1.0)
        self.vx = float(out[0] * speed)
        self.vy = float(out[1] * speed)

        # add border push (prey only, but can do for predators too)
        bx, by = push_from_border(self)
        self.vx += bx
        self.vy += by

        # update position
        self.x += self.vx
        self.y += self.vy
        self.x = max(0, min(self.W, self.x))
        self.y = max(0, min(self.H, self.y))

        return inp, h, out


    def learn(self, inp, h, out, target):
        self.w1,self.b1,self.w2,self.b2 = mlp_back(
            inp,h,out,target,self.w1,self.b1,self.w2,self.b2,self.lr
        )

    def draw(self):
        pygame.draw.circle(screen, self.color,
                           (int(self.x), int(self.y)), int(self.size))

prey = [Agent((80,180,255),"prey",W,H) for _ in range(20)]
pred = [Agent((255,90,80),"pred",W,H) for _ in range(6)]

running=True
while running:
    for e in pygame.event.get():
        if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
            running=False

    mx,my = pygame.mouse.get_pos()
    screen.fill((12,12,20))

    # predators
    import pygame, random, math, numpy as np
    pygame.init()
    PRED_SIGHT_RADIUS = 200  # pixels

    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    W, H = screen.get_size()
    clock = pygame.time.Clock()

    def neighbour_offset(agent, group, k=3):
        others = [a for a in group if a is not agent]
        if not others:
            return 0.0, 0.0
        # sort by distance
        nearest = sorted(others, key=lambda o:(o.x-agent.x)**2+(o.y-agent.y)**2)[:k]
        dx = sum(o.x - agent.x for o in nearest) / max(1,len(nearest))
        dy = sum(o.y - agent.y for o in nearest) / max(1,len(nearest))
        return dx, dy

    def normalize(v):
        d = math.hypot(v[0], v[1]) + 1e-6
        return np.array([v[0]/d, v[1]/d])

    def mlp_fwd(x, w1, b1, w2, b2):
        h = np.tanh(x @ w1 + b1)
        o = np.tanh(h @ w2 + b2)
        return h, o

    def mlp_back(x, h, o, target, w1, b1, w2, b2, lr):
        go = (o - target) * (1 - o**2)
        gw2 = np.outer(h, go)
        gb2 = go
        gh  = (go @ w2.T) * (1 - h**2)
        gw1 = np.outer(x, gh)
        gb1 = gh
        w1 -= lr*gw1; b1 -= lr*gb1
        w2 -= lr*gw2; b2 -= lr*gb2
        return w1,b1,w2,b2


    prey = [Agent((80,180,255),"prey",W,H) for _ in range(20)]
    pred = [Agent((255,90,80),"pred",W,H) for _ in range(6)]

    running=True
    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                running=False

        screen.fill((12,12,20))


        # predators
        for p in pred:
            p.vy, p.vx = 0, 0
            prev_x, prev_y = p.x, p.y
            distance_moved = math.hypot(p.x - prev_x, p.y - prev_y)

            targ = 0
            border_push = push_from_border(p)   # or p for predator

            p.vx += border_push[0]
            p.vy += border_push[1]

            # get prey within sight radius
            visible_prey = [f for f in prey if math.hypot(f.x - p.x, f.y - p.y) <= PRED_SIGHT_RADIUS]
            if visible_prey:
                nearest = min(visible_prey, key=lambda f: (f.x - p.x)**2 + (f.y - p.y)**2)
                tpos = (nearest.x, nearest.y)
                # inside prey loop
                neigh_vecs = neighbour_vectors(p, prey, p.k, none = True)
                print(visible_prey)
                inp, h, out = p.forward_and_move((visible_prey[0].x, visible_prey[0].x), neigh_vecs)


                d = math.hypot(p.x - nearest.x, p.y - nearest.y)
                eaten = d < p.size + 5

                if eaten:
                    p.score += 1
                    p.size += 1.4
                    p.speed_factor *= 0.92
                    nearest.respawn()
                    targ = normalize((nearest.x - p.x, nearest.y - p.y)) * 3
                else:
                    targ = normalize((nearest.x - p.x, nearest.y - p.y)) * 0.6

                # apply border penalty

                p.learn(inp, h, out, targ)

                # natural shrinking
                p.size = max(6, p.size - 0.003)
                p.speed_factor = max(0.4, p.speed_factor + 0.0003)

                p.draw()
            else:
                # encourage predator to roam
                                # before updating position
                prev_pos = np.array([p.x, p.y])
                # inside prey loop
                neigh_vecs = neighbour_vectors(f, prey, f.k)
                inp, h, out = f.forward_and_move((nearest_pred.x, nearest_pred.y), neigh_vecs)
                movement = np.array([p.vx, p.vy])              # actual movement including border push
                targ = normalize(movement) * 0.5               # scale reward for roaming
                p.learn(inp, h, out, targ)

                p.draw()


        # prey
        for f in prey:
            target = 0

            f.vy, f.vx = 0, 0

            border_push = push_from_border(f)   # or p for predator
            #targ += border_push
            f.vx += border_push[0]
            f.vy += border_push[1]

            nearest_pred = [p for p in pred if math.hypot(p.x - f.x, p.y - f.y)]

            ndx, ndy = neighbour_offset(f, prey)
            # inside prey loop
            neigh_vecs = neighbour_vectors(f, prey, f.k)

            inp, h, out = f.forward_and_move((nearest_pred[0].x, nearest_pred[0].y), neigh_vecs)

            dist = math.hypot(f.x - nearest_pred.x, f.y - nearest_pred.y)
            eaten_pred = dist < nearest_pred.size + 5
            eaten_mouse = math.hypot(f.x - mx, f.y - my) < 18

            if eaten_pred:
                f.score -= 1
                targ = -normalize((nearest_pred.x - f.x, nearest_pred.y - f.y)) * 3
                f.respawn()
            else:
                targ = -normalize((nearest_pred.x - f.x, nearest_pred.y - f.y)) * 0.8

            # mouse penalty
            if eaten_mouse:
                f.score -= 1
                targ_mouse = -normalize((mx - f.x, my - f.y)) * 3
                f.respawn()
                f.learn(inp, h, out, targ_mouse)  # separate learn for mouse event

            f.learn(inp, h, out, targ)
            f.draw()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


    # prey
    for f in prey:
        nearest_pred = min(pred, key=lambda q:(q.x-f.x)**2+(q.y-f.y)**2)
        ndx, ndy = neighbour_offset(f, prey)   # new line

        # inside prey loop
        neigh_vecs = neighbour_vectors(f, prey, f.k)
        inp, h, out = f.forward_and_move((nearest_pred.x, nearest_pred.y), neigh_vecs)


        dist = math.hypot(f.x-nearest_pred.x, f.y-nearest_pred.y)
        eaten_pred = dist < nearest_pred.size + 5
        eaten_mouse = math.hypot(f.x-mx,f.y-my)<18

        if eaten_pred:
            f.score -= 1
            targ = -normalize((nearest_pred.x-f.x, nearest_pred.y-f.y))*3
            f.respawn()
        else:
            targ = -normalize((nearest_pred.x-f.x, nearest_pred.y-f.y))*0.8

        if eaten_mouse:
            f.score -= 1
            targ_mouse = -normalize((mx-f.x, my-f.y))*3
            f.respawn()
            f.learn(inp,h,out,targ_mouse)

        f.learn(inp,h,out,targ)
        f.draw()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
