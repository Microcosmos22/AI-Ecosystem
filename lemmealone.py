import pygame, random, math, numpy as np
pygame.init()

screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
W, H = screen.get_size()
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

def normalize(v):
    d = math.hypot(v[0], v[1]) + 1e-6
    return np.array([v[0]/d, v[1]/d])

def mlp_fwd(x, w1, b1, w2, b2):
    h = np.tanh(x @ w1 + b1)
    o = np.tanh(h @ w2 + b2)
    return h, o

def mlp_back(x, h, o, target, w1, b1, w2, b2, lr):
    # simple MSE backprop for tiny net
    go = (o - target) * (1 - o**2)
    gw2 = np.outer(h, go)
    gb2 = go
    gh = (go @ w2.T) * (1 - h**2)
    gw1 = np.outer(x, gh)
    gb1 = gh
    w1 -= lr * gw1; b1 -= lr * gb1
    w2 -= lr * gw2; b2 -= lr * gb2
    return w1, b1, w2, b2

class Agent:
    def __init__(self, color, kind, W, H):
        self.kind = kind
        self.color = color
        self.W = W; self.H = H
        self.respawn()
        # START WITH ZERO WEIGHTS (no prior policy)
        self.w1 = np.zeros((2, 6))
        self.b1 = np.zeros(6)
        self.w2 = np.zeros((6, 2))
        self.b2 = np.zeros(2)
        self.lr = 0.02 if kind == "pred" else 0.015
        self.explore = 1.2  # exploration noise amplitude (decays)
        self.score = 0

    def respawn(self):
        self.x = random.random() * self.W
        self.y = random.random() * self.H
        self.vx = self.vy = 0

    def forward_and_move(self, tx, ty):
        dx = (tx - self.x) / self.W
        dy = (ty - self.y) / self.H
        inp = np.array([dx, dy])
        h, out = mlp_fwd(inp, self.w1, self.b1, self.w2, self.b2)

        # exploration: essential because zero weights => zero output
        noise = np.random.randn(2) * self.explore
        out_noisy = out + noise

        # small decay of exploration over time
        self.explore *= 0.9998

        self.vx, self.vy = out_noisy * 6
        self.x += self.vx
        self.y += self.vy

        # keep inside screen box
        self.x = max(0, min(self.W, self.x))
        self.y = max(0, min(self.H, self.y))

        return inp, h, out

    def learn(self, inp, h, out, target):
        self.w1, self.b1, self.w2, self.b2 = mlp_back(
            inp, h, out, target, self.w1, self.b1, self.w2, self.b2, self.lr
        )

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), 8)

# create agents
prey = [Agent((80,180,255), "prey", W, H) for _ in range(20)]
pred = [Agent((230,90,80), "pred", W, H) for _ in range(7)]

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            running = False

    mx, my = pygame.mouse.get_pos()
    screen.fill((12,12,20))

    # predators act: try to move toward nearest prey; rewarded for eating
    for p in pred:
        if not prey: break
        nearest = min(prey, key=lambda f: (f.x - p.x)**2 + (f.y - p.y)**2)
        inp, h, out = p.forward_and_move(nearest.x, nearest.y)
        dist = math.hypot(p.x - nearest.x, p.y - nearest.y)
        eaten = dist < 18
        if eaten:
            p.score += 1                 # predator gains point
            # strong positive target: point toward prey direction
            targ = normalize((nearest.x - p.x, nearest.y - p.y)) * 3.0
            nearest.respawn()
        else:
            # weaker target: encourage moving toward prey
            targ = normalize((nearest.x - p.x, nearest.y - p.y)) * 0.6
        p.learn(inp, h, out, targ)
        p.draw(screen)

    # prey act: learn to flee from nearest predator and from mouse (mouse eats prey)
    for f in prey:
        nearest_pred = min(pred, key=lambda q: (q.x - f.x)**2 + (q.y - f.y)**2)
        # combine influences: primary is predator, secondary is mouse position
        # use predator as target to flee from
        inp, h, out = f.forward_and_move(nearest_pred.x, nearest_pred.y)
        dist_pred = math.hypot(f.x - nearest_pred.x, f.y - nearest_pred.y)
        eaten_by_pred = dist_pred < 18
        # mouse eat check
        eaten_by_mouse = math.hypot(f.x - mx, f.y - my) < 25

        if eaten_by_pred:
            f.score -= 1                # prey loses point when eaten by predator
            targ = -normalize((nearest_pred.x - f.x, nearest_pred.y - f.y)) * 3.0
            f.respawn()
        else:
            # encourage fleeing (away from predator)
            targ = -normalize((nearest_pred.x - f.x, nearest_pred.y - f.y)) * 0.8

        if eaten_by_mouse:
            f.score -= 1                # prey loses point when eaten by mouse
            # strong flee target from mouse and respawn
            targ_mouse = -normalize((mx - f.x, my - f.y)) * 3.0
            f.respawn()
            # learn from mouse-eaten event too
            f.learn(inp, h, out, targ_mouse)
        else:
            f.learn(inp, h, out, targ)

        f.draw(screen)

    # Draw simple HUD: total scores
    total_pred_score = sum(p.score for p in pred)
    total_prey_score = sum(f.score for f in prey)
    hud1 = font.render(f"Predators score: {total_pred_score}", True, (255,200,200))
    hud2 = font.render(f"Prey score: {total_prey_score}", True, (200,230,255))
    screen.blit(hud1, (12, 12))
    screen.blit(hud2, (12, 40))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
