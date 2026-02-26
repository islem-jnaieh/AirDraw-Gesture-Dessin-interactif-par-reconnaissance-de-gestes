import cv2
import numpy as np
import mediapipe as mp
import os
import time
import threading
from queue import Queue

# ============= CONFIGURATION =============
class Config:
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    MIN_DETECTION_CONFIDENCE = 0.8
    MIN_TRACKING_CONFIDENCE = 0.7
    ROBOT_SIZE = 35
    ROBOT_COLOR = (30, 180, 255)  # Orange-bleu
    GOAL_SIZE = 30
    GOAL_COLOR = (50, 255, 150)  # Vert clair
    OBSTACLE_COLOR = (100, 100, 200)  # Bleu-gris

# ============= INITIALISATION =============
def init_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
    )
    mp_draw = mp.solutions.drawing_utils
    return hands, mp_hands, mp_draw

def init_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
    return cap

# ============= ROBOT STATE MACHINE =============
class RobotState:
    """Énumération des états du robot (FSM)"""
    IDLE = 0
    MOVE_RIGHT = 1
    MOVE_LEFT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4
    COLLISION = 5
    VICTORY = 6

# ============= CLASSE ROBOT =============
class Robot:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.x = 50
        self.y = h // 2
        self.size = Config.ROBOT_SIZE
        self.speed = 5
        # Machine à états (FSM)
        self.state = RobotState.IDLE
        
    def set_state(self, new_state):
        """Changer l'état du robot"""
        self.state = new_state
    
    def update(self):
        """Mettre à jour la position du robot selon son état"""
        if self.state == RobotState.MOVE_RIGHT:
            self.x = min(self.x + self.speed, self.w - self.size)
        elif self.state == RobotState.MOVE_LEFT:
            self.x = max(self.x - self.speed, 0)
        elif self.state == RobotState.MOVE_UP:
            self.y = max(self.y - self.speed, 50)
        elif self.state == RobotState.MOVE_DOWN:
            self.y = min(self.y + self.speed, self.h - self.size)
        elif self.state == RobotState.IDLE or self.state == RobotState.COLLISION or self.state == RobotState.VICTORY:
            # Pas de mouvement dans ces états
            pass
        
        # Revenir à IDLE après action
        if self.state in [RobotState.MOVE_RIGHT, RobotState.MOVE_LEFT, RobotState.MOVE_UP, RobotState.MOVE_DOWN]:
            self.state = RobotState.IDLE
    
    def move_right(self):
        """Héritage pour compatibilité (déprécié - utiliser set_state)"""
        self.x = min(self.x + self.speed, self.w - self.size)
    
    def move_left(self):
        """Héritage pour compatibilité (déprécié - utiliser set_state)"""
        self.x = max(self.x - self.speed, 0)
    
    def move_up(self):
        """Héritage pour compatibilité (déprécié - utiliser set_state)"""
        self.y = max(self.y - self.speed, 50)
    
    def move_down(self):
        """Héritage pour compatibilité (déprécié - utiliser set_state)"""
        self.y = min(self.y + self.speed, self.h - self.size)
    
    def draw(self, canvas):
        x, y = int(self.x), int(self.y)
        s = int(self.size)
        
        # Ombre
        cv2.rectangle(canvas, (x+2, y+2), (x+s+2, y+s+2),
                     (50, 50, 50), -1)
        
        # Corps du robot avec couleur vibrante
        cv2.rectangle(canvas, (x, y), (x+s, y+s),
                     Config.ROBOT_COLOR, -1)
        
        # Bordure highlight
        cv2.rectangle(canvas, (x, y), (x+s, y+s),
                     (200, 220, 255), 3)
        
        # Petit accent interne
        cv2.circle(canvas, (x+s//3, y+s//3), 4, (255, 255, 255), -1)
    
    def check_collision_rect(self, x1, y1, x2, y2):
        """Vérifier collision avec un rectangle"""
        robot_center_x = self.x + self.size / 2
        robot_center_y = self.y + self.size / 2
        
        return (x1 <= robot_center_x <= x2 and y1 <= robot_center_y <= y2)


# ============= CLASSE NIVEAU =============
class Level:
    def __init__(self, level_num, w, h):
        self.level_num = level_num
        self.w = w
        self.h = h
        # Position du but change à chaque niveau (plus proche avec le temps)
        self.goal_x = w - 80 - level_num * 5
        self.goal_y = h // 2
        # La cible devient plus petite avec les niveaux
        self.goal_size = max(20, Config.GOAL_SIZE - level_num)
        self.obstacles = []
        self.reached = False
        
        # Générer les obstacles basés sur le niveau
        self._generate_obstacles()
    
    def _generate_obstacles(self):
        """Générer les obstacles selon la difficulté"""
        import random
        random.seed(self.level_num)  # Seed différent par niveau
        
        # Augmenter les obstacles de manière progressive
        num_obstacles = min(2 + self.level_num * 2, 12)  # 2 -> 4 -> 6 -> 8 -> 10 -> 12...
        
        for i in range(num_obstacles):
            # S'assurer que l'obstacle ne chevauche pas le but
            valid = False
            attempts = 0
            while not valid and attempts < 10:
                # Les obstacles grossissent avec les niveaux
                obs_w = 35 + self.level_num * 8
                obs_h = 35 + self.level_num * 8
                
                # Disperser sur TOUTE la zone de jeu
                # Laisser 80 pixels à gauche pour la position de départ du robot
                # Laisser 100 pixels à droite pour le but
                obs_x = random.randint(80, self.w - obs_w - 100)
                obs_y = random.randint(50, self.h - obs_h - 20)
                
                # Vérifier que l'obstacle ne chevauche pas le but
                goal_buffer = 60  # Zone autour du but
                if not (obs_x < self.goal_x + self.goal_size + goal_buffer and 
                        obs_x + obs_w > self.goal_x - goal_buffer and
                        obs_y < self.goal_y + self.goal_size + goal_buffer and
                        obs_y + obs_h > self.goal_y - goal_buffer):
                    valid = True
                
                attempts += 1
            
            if valid:
                self.obstacles.append({
                    'x': obs_x,
                    'y': obs_y,
                    'w': obs_w,
                    'h': obs_h
                })
    
    def check_goal(self, robot):
        """Vérifier si le robot a atteint le but"""
        dist = np.sqrt((robot.x - self.goal_x)**2 + (robot.y - self.goal_y)**2)
        self.reached = dist < (robot.size / 2 + self.goal_size / 2)
        return self.reached
    
    def check_collision(self, robot):
        """Vérifier collision avec les obstacles"""
        for obs in self.obstacles:
            if robot.check_collision_rect(obs['x'], obs['y'], 
                                         obs['x'] + obs['w'], 
                                         obs['y'] + obs['h']):
                return True
        return False
    
    def draw(self, canvas):
        """Dessiner le niveau"""
        # But (objectif)
        gx, gy = int(self.goal_x), int(self.goal_y)
        gs = int(self.goal_size)
        
        # Ombre du but
        cv2.rectangle(canvas, (gx+2, gy+2), (gx+gs+2, gy+gs+2),
                     (30, 30, 30), -1)
        
        # Corps du but avec dégradé
        cv2.rectangle(canvas, (gx, gy), (gx+gs, gy+gs),
                     Config.GOAL_COLOR, -1)
        
        # Effet brillant sur le but
        for j in range(gs // 3):
            cv2.line(canvas, (gx+j, gy), (gx+j+2, gy+gs),
                    (200, 255, 200), 1)
        
        # Bordure du but avec couleur plus sombre
        cv2.rectangle(canvas, (gx, gy), (gx+gs, gy+gs),
                     (0, 150, 100), 3)
        
        # Etoile sur le but
        cv2.circle(canvas, (gx+gs//2, gy+gs//2), 3, (255, 255, 100), -1)
        
        # Obstacles avec style amélioré
        for obs in self.obstacles:
            ox, oy = int(obs['x']), int(obs['y'])
            ow, oh = int(obs['w']), int(obs['h'])
            
            # Ombre
            cv2.rectangle(canvas, (ox+3, oy+3), (ox+ow+3, oy+oh+3),
                         (20, 20, 20), -1)
            
            # Corps de l'obstacle avec dégradé
            # Dessiner le rectangle principal
            cv2.rectangle(canvas, (ox, oy), (ox+ow, oy+oh),
                         Config.OBSTACLE_COLOR, -1)
            
            # Ajouter un dégradé léger (effet 3D)
            for j in range(oh // 3):
                color_intensity = 255 - (j * 15)
                alpha = 0.1
                overlay = canvas.copy()
                cv2.line(overlay, (ox, oy+j), (ox+ow, oy+j),
                        (min(255, Config.OBSTACLE_COLOR[0]+30), 
                         min(255, Config.OBSTACLE_COLOR[1]+30), 
                         min(255, Config.OBSTACLE_COLOR[2]+30)), 1)
                cv2.addWeighted(overlay, alpha, canvas, 1-alpha, 0, canvas)
            
            # Bordure colorée
            cv2.rectangle(canvas, (ox, oy), (ox+ow, oy+oh),
                         (180, 180, 100), 2)
            
            # Petits carrés décoration
            cv2.rectangle(canvas, (ox+4, oy+4), (ox+8, oy+8),
                         (200, 200, 150), -1)
            cv2.rectangle(canvas, (ox+ow-8, oy+4), (ox+ow-4, oy+8),
                         (200, 200, 150), -1)
            cv2.rectangle(canvas, (ox+4, oy+oh-8), (ox+8, oy+oh-4),
                         (200, 200, 150), -1)
            cv2.rectangle(canvas, (ox+ow-8, oy+oh-8), (ox+ow-4, oy+oh-4),
                         (200, 200, 150), -1)
        
        # Titre du niveau avec style
        level_text = "NIVEAU " + str(self.level_num)
        cv2.putText(canvas, level_text, (12, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        cv2.putText(canvas, level_text, (10, 33),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 100), 2)


# ============= DÉTECTION GESTES =============
def count_fingers(lm_list):
    """Compter les doigts levés (simplifié)"""
    if len(lm_list) < 21:
        return 0
    
    fingers = 0
    # Index
    if lm_list[8][1] < lm_list[6][1]:
        fingers += 1
    # Majeur
    if lm_list[12][1] < lm_list[10][1]:
        fingers += 1
    # Annulaire
    if lm_list[16][1] < lm_list[14][1]:
        fingers += 1
    # Auriculaire
    if lm_list[20][1] < lm_list[18][1]:
        fingers += 1
    
    return fingers


def is_single_index(lm_list):
    """Index levé seul"""
    if len(lm_list) < 21:
        return False
    index_up = lm_list[8][1] < lm_list[6][1]
    majeur_down = lm_list[12][1] >= lm_list[10][1]
    return index_up and majeur_down


def is_index_majeur(lm_list):
    """Index + Majeur levés"""
    if len(lm_list) < 21:
        return False
    index_up = lm_list[8][1] < lm_list[6][1]
    majeur_up = lm_list[12][1] < lm_list[10][1]
    annulaire_down = lm_list[16][1] >= lm_list[14][1]
    return index_up and majeur_up and annulaire_down


def is_three_fingers(lm_list):
    """3 doigts levés (index + majeur + annulaire)"""
    if len(lm_list) < 21:
        return False
    index_up = lm_list[8][1] < lm_list[6][1]
    majeur_up = lm_list[12][1] < lm_list[10][1]
    annulaire_up = lm_list[16][1] < lm_list[14][1]
    auriculaire_down = lm_list[20][1] >= lm_list[18][1]
    return index_up and majeur_up and annulaire_up and auriculaire_down


def is_fist(lm_list):
    """Poing (aucun doigt levé)"""
    return count_fingers(lm_list) == 0


def is_four_fingers(lm_list):
    """4 doigts levés (tous sauf le pouce)"""
    return count_fingers(lm_list) == 4


def draw_legend(canvas):
    """Afficher la légende des gestes"""
    h, w = canvas.shape[:2]
    
    # Fond semi-transparent blanc
    overlay = canvas.copy()
    cv2.rectangle(overlay, (w - 280, 120), (w - 10, 300), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
    
    # Bordure
    cv2.rectangle(canvas, (w - 280, 120), (w - 10, 300), (0, 0, 0), 2)
    
    # Textes
    y_start = 135
    line_height = 30
    
    cv2.putText(canvas, "CONTROLES:", (w - 270, y_start), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(canvas, "-> Index: DROITE", (w - 270, y_start + line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    
    cv2.putText(canvas, "-> 2 doigts: GAUCHE", (w - 270, y_start + line_height * 2), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    
    cv2.putText(canvas, "-> 3 doigts: HAUT", (w - 270, y_start + line_height * 3), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    
    cv2.putText(canvas, "-> 4 doigts: BAS", (w - 270, y_start + line_height * 4), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    
    cv2.putText(canvas, "-> Poing: STOP", (w - 270, y_start + line_height * 5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 1) 

def draw_legend_on_frame(frame):
    """Afficher la légende des gestes sur le frame"""
    h, w = frame.shape[:2]
    
    # Fond avec bordure dégradée - plus compact
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (250, 185), (200, 150, 80), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Bordure décorée
    cv2.rectangle(frame, (5, 5), (250, 185), (100, 200, 255), 2)
    cv2.rectangle(frame, (7, 7), (248, 183), (200, 255, 200), 1)
    
    # Titre plus petit (en français)
    cv2.putText(frame, "COMMANDES", (12, 22), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 50, 50), 1)
    
    # Commandes avec couleur - plus compactes
    y_start = 40
    line_height = 28
    
    # Garder les libellés en français et afficher le nombre de doigts
    # Utiliser uniquement des symboles ASCII pour éviter les '???' si la fonte Unicode manque
    commands = [
        ("1 doigt : DROITE", "->", (50, 150, 255)),
        ("2 doigts : GAUCHE", "<-", (50, 255, 200)),
        ("3 doigts : HAUT", "^", (255, 200, 100)),
        ("4 doigts : BAS", "v", (255, 150, 100)),
        ("0 doigt : STOP", "STOP", (255, 100, 100))
    ]
    
    for idx, (gesture, action, color) in enumerate(commands):
        y = y_start + line_height * idx
        # Texte du geste (inclut le nombre de doigts)
        cv2.putText(frame, gesture, (12, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 50), 1)
        # Symbole action coloré (ASCII)
        cv2.putText(frame, action, (175, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def extract_landmarks(hand_landmarks, w, h):
    return [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks.landmark]

# ============= RTOS SCHEDULER (Simulation) =============
class Task:
    """Représente une tâche périodique dans le scheduler RTOS"""
    def __init__(self, name, func, period_ms):
        self.name = name
        self.func = func
        self.period_ms = period_ms  # Période en millisecondes
        self.last_exec_time = time.perf_counter()
    
    def should_run(self, current_time):
        """Vérifier si la tâche doit s'exécuter"""
        elapsed = (current_time - self.last_exec_time) * 1000  # Convertir en ms
        return elapsed >= self.period_ms
    
    def execute(self, current_time, *args, **kwargs):
        """Exécuter la tâche et mettre à jour le temps d'exécution"""
        self.func(*args, **kwargs)
        self.last_exec_time = current_time

class Scheduler:
    """Mini-RTOS : Scheduler pour gérer l'exécution périodique des tâches"""
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        """Ajouter une tâche au scheduler"""
        self.tasks.append(task)
    
    def run(self, current_time):
        """Exécuter toutes les tâches prêtes"""
        for task in self.tasks:
            if task.should_run(current_time):
                task.execute(current_time)

# ============= THREAD CAMERA ============
def camera_capture(frame_queue):
    cap = init_camera()
    while True:
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.flip(frame,1)
        try:
            frame_queue.put_nowait(frame)
        except:
            pass
    cap.release()

# ============= THREAD MEDIAPIPE ==========
def mediapipe_detection(hands, frame_queue, result_queue):
    while True:
        try:
            frame = frame_queue.get_nowait()
        except:
            time.sleep(0.001)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        try:
            result_queue.put_nowait((frame,result))
        except:
            pass

def create_canvas(w, h):
    """Créer ou réinitialiser le canvas avec un fond gradué"""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 240
    # Ajouter un fond gradué subtil
    for i in range(h):
        canvas[i, :] = [240 - i//4, 240 - i//8, 240]
    return canvas


# ============= BOUCLE PRINCIPALE ==========
def main():
    hands, mp_hands, mp_draw = init_mediapipe()
    
    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)
    threading.Thread(target=camera_capture, args=(frame_queue,), daemon=True).start()
    threading.Thread(target=mediapipe_detection, args=(hands, frame_queue, result_queue), daemon=True).start()
    
    canvas = None
    prev_x, prev_y = 0, 0
    color_index = 0
    save_counter = 1
    
    # Robot et niveau
    robot = None
    level = None
    level_num = 1
    
    # FPS
    frame_count = 0
    fps_display = "30.0"
    last_fps_time = time.perf_counter()
    
    # État du geste
    last_gesture = ""
    gesture_timer = 0
    
    # Écran victoire
    victory_timer = 0
    
    print("=== ROBOT QUEST ===")
    print("Guidez le carré rouge jusqu'au carré vert !")
    print("Q: quitter")
    print("==================\n")
    
    while True:
        try:
            frame, result = result_queue.get_nowait()
        except:
            time.sleep(0.001)
            continue
        
        h, w, c = frame.shape
        if canvas is None:
            # Créer un template de fond et l'utiliser pour générer un canvas propre chaque frame
            background_template = create_canvas(w, h)
            canvas = background_template.copy()
            robot = Robot(w, h)
            level = Level(level_num, w, h)
            
            # Initialiser le scheduler RTOS
            scheduler = Scheduler()
            # TaskRobotControl: Mettre à jour l'état du robot (5ms = 200Hz)
            task_control = Task("RobotControl", robot.update, 5)
            scheduler.add_task(task_control)

        # Recréer le canvas à partir du template pour éviter les traces du robot
        canvas = background_template.copy()
        
        # --- Afficher la légende sur le frame ---
        draw_legend_on_frame(frame)
        
        # --- Calcul FPS ---
        frame_count += 1
        current_time = time.perf_counter()
        elapsed = current_time - last_fps_time
        if elapsed >= 1.0:
            fps_display = f"{frame_count / elapsed:.1f}"
            frame_count = 0
            last_fps_time = current_time
        
        # --- Traitement main et gestes ---
        if result.multi_hand_landmarks and victory_timer <= 0:
            for hand_landmarks in result.multi_hand_landmarks:
                lm_list = extract_landmarks(hand_landmarks, w, h)
                
                # Détection gestes pour décider de l'état du robot
                if is_single_index(lm_list):
                    robot.set_state(RobotState.MOVE_RIGHT)
                    last_gesture = "DROITE"
                    gesture_timer = 15
                elif is_index_majeur(lm_list):
                    robot.set_state(RobotState.MOVE_LEFT)
                    last_gesture = "GAUCHE"
                    gesture_timer = 15
                elif is_three_fingers(lm_list):
                    robot.set_state(RobotState.MOVE_UP)
                    last_gesture = "HAUT"
                    gesture_timer = 15
                elif is_four_fingers(lm_list):
                    robot.set_state(RobotState.MOVE_DOWN)
                    last_gesture = "BAS"
                    gesture_timer = 15
                elif is_fist(lm_list):
                    robot.set_state(RobotState.IDLE)
                    last_gesture = "STOP"
                    gesture_timer = 15
                
                # Dessin classique avec la palette en haut
                if len(lm_list) > 8:
                    index_x, index_y = lm_list[8]
                    # Just control the robot, no drawing
                
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- Exécuter la FSM du robot (mise à jour état) ---
        robot.update()
        
        # --- Dessiner le niveau et le robot ---
        level.draw(canvas)
        robot.draw(canvas)
        
        # --- Vérifier collision avec les obstacles ---
        if level.check_collision(robot):
            # Collision, réinitialiser complètement le niveau et le template
            background_template = create_canvas(w, h)
            canvas = background_template.copy()
            level = Level(level_num, w, h)
            robot = Robot(w, h)
            print(f"💥 Collision! Niveau {level_num} réinitialisé.")
        
        # --- Vérifier si victoire ---
        if level.check_goal(robot) and victory_timer <= 0:
            victory_timer = 120  # 4 secondes à 30 FPS
            print(f"✓ NIVEAU {level.level_num} TERMINÉ!")

        # --- Affichage écran victoire ---
        if victory_timer > 0:
            victory_timer -= 1
            # Fond gradient doré avec transparence
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (50, 180, 220), -1)
            intensity = 0.4 + 0.2 * (victory_timer % 60 / 60.0)
            cv2.addWeighted(overlay, intensity, canvas, 1 - intensity, 0, canvas)
            
            # Boîte de victoire
            box_h = 120
            box_y = (h - box_h) // 2
            box_padding = 30
            box_x = box_padding
            box_w = w - box_padding * 2
            
            # Fond de la boîte
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         (40, 150, 200), -1)
            # Bordure brillante
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         (200, 255, 150), 3)
            
            # Texte VICTOIRE
            # Texte victoire sans caractère Unicode pour éviter les '???'
            vic_text = "NIVEAU " + str(level.level_num) + " TERMINE!"
            # Texte victoire plus petit pour ne pas dominer l'écran
            vic_scale = 1.2
            vic_thick = 2
            text_size = cv2.getTextSize(vic_text, cv2.FONT_HERSHEY_SIMPLEX, vic_scale, vic_thick)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2 - 10
            cv2.putText(canvas, vic_text, (text_x+1, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, vic_scale, (0, 0, 0), max(1, vic_thick-1))
            cv2.putText(canvas, vic_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, vic_scale, (100, 255, 100), vic_thick)

            # Message secondaire plus discret
            msg = "Prochain niveau..."
            msg_scale = 0.75
            msg_thick = 1
            msg_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, msg_scale, msg_thick)[0]
            msg_x = (w - msg_size[0]) // 2
            msg_y = h // 2 + 35
            cv2.putText(canvas, msg, (msg_x, msg_y), cv2.FONT_HERSHEY_SIMPLEX, msg_scale, (255, 255, 100), msg_thick)
            
            # Auto-advance au prochain niveau
            if victory_timer == 1:
                level_num += 1
                # Recréer le template et canvas pour le nouveau niveau
                background_template = create_canvas(w, h)
                canvas = background_template.copy()
                level = Level(level_num, w, h)
                robot = Robot(w, h)
                victory_timer = 0

        cv2.putText(frame, f"FPS: {fps_display}", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        combined = np.hstack((frame, canvas))
        cv2.imshow("Virtual Paint + Robot Quest", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Fermeture...")
            break
        elif key == ord('s'):
            save_path = os.path.join(os.getcwd(), f"drawing_{save_counter}.png")
            cv2.imwrite(save_path, canvas)
            print(f"✓ Dessin sauvegardé : {save_path}")
            save_counter += 1
        elif key == ord('l'):
            if robot and robot.log:
                log_path = os.path.join(os.getcwd(), f"robot_log_{int(time.time())}.txt")
                with open(log_path, 'w') as f:
                    for entry in robot.log:
                        f.write(entry + '\n')
                print(f"✓ Log sauvegardé : {log_path}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()