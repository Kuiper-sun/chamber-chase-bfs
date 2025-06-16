from flask import Flask, render_template, jsonify, request
import random
from collections import deque
import math

app = Flask(__name__)

class ChamberChaseGame:
    def __init__(self, width=12, height=12, difficulty='medium'):
        self.width = width
        self.height = height
        self.difficulty = difficulty
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.player_pos = None
        self.ai_pos = None
        self.exit_positions = []  # Multiple exits
        self.game_over = False
        self.game_won = False
        self.turn = 'player'
        self.ai_prediction_depth = self.get_ai_prediction_depth()
        self.move_count = 0
        self.max_moves = self.get_max_moves()
        self.safe_distance = self.get_safe_distance()
        
        # Vision system
        self.player_vision_range = self.get_player_vision_range()
        self.ai_vision_range = self.get_ai_vision_range()
        self.player_detected = False
        self.last_known_player_pos = None
        self.detection_cooldown = 0
        
        # Improved AI system
        self.ai_state = 'patrol'  # 'patrol', 'investigate', 'chase'
        self.patrol_points = []
        self.current_patrol_target = 0
        self.investigation_target = None
        self.turns_since_player_seen = 0
        self.ai_path_history = deque(maxlen=5)  # Track recent positions
        
        # Multiple exits
        self.num_exits = self.get_num_exits()
        
        self.initialize_game()
    
    def get_ai_prediction_depth(self):
        """Get AI prediction depth based on difficulty"""
        if self.difficulty == 'easy':
            return 2
        elif self.difficulty == 'medium':
            return 3
        elif self.difficulty == 'hard':
            return 4
        else:  # nightmare
            return 5
    
    def get_player_vision_range(self):
        """Get player vision range based on difficulty"""
        if self.difficulty == 'easy':
            return 4
        elif self.difficulty == 'medium':
            return 3
        elif self.difficulty == 'hard':
            return 2
        else:  # nightmare
            return 2
    
    def get_ai_vision_range(self):
        """Get AI vision range based on difficulty"""
        if self.difficulty == 'easy':
            return 2
        elif self.difficulty == 'medium':
            return 3
        elif self.difficulty == 'hard':
            return 4
        else:  # nightmare
            return 5
    
    def get_num_exits(self):
        """Get number of exits based on difficulty"""
        if self.difficulty == 'easy':
            return 3
        elif self.difficulty == 'medium':
            return 2
        elif self.difficulty == 'hard':
            return 2
        else:  # nightmare
            return 1
    
    def get_max_moves(self):
        """Get maximum moves based on optimal path length"""
        if not self.exit_positions or not self.player_pos:
            return 50
            
        # Calculate minimum distance to any exit
        min_distance = min(self.bfs_distance(self.player_pos, exit_pos) 
                          for exit_pos in self.exit_positions)
        
        if self.difficulty == 'easy':
            return min_distance * 4
        elif self.difficulty == 'medium':
            return min_distance * 3
        elif self.difficulty == 'hard':
            return int(min_distance * 2.5)
        else:  # nightmare
            return min_distance * 2
    
    def get_safe_distance(self):
        """Minimum safe distance between player and AI at start"""
        if self.difficulty == 'easy':
            return max(5, (self.width + self.height) // 4)
        elif self.difficulty == 'medium':
            return max(4, (self.width + self.height) // 5)
        elif self.difficulty == 'hard':
            return max(3, (self.width + self.height) // 6)
        else:  # nightmare
            return 3
    
    def get_obstacle_density(self):
        """Get obstacle density based on difficulty"""
        total_tiles = self.width * self.height
        if self.difficulty == 'easy':
            return int(total_tiles * 0.15)
        elif self.difficulty == 'medium':
            return int(total_tiles * 0.20)
        elif self.difficulty == 'hard':
            return int(total_tiles * 0.25)
        else:  # nightmare
            return int(total_tiles * 0.28)
    
    def bfs_distance(self, start, end):
        """Calculate shortest path distance using BFS"""
        if start == end:
            return 0
        
        queue = deque([(start, 0)])
        visited = set([start])
        
        while queue:
            current_pos, distance = queue.popleft()
            
            for neighbor in self.get_neighbors(current_pos[0], current_pos[1]):
                if neighbor not in visited:
                    if neighbor == end:
                        return distance + 1
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return float('inf')
    
    def bfs_pathfinding(self, start_pos, target_pos):
        """BFS pathfinding algorithm - returns next move"""
        if start_pos == target_pos:
            return start_pos
        
        queue = deque([(start_pos, [start_pos])])
        visited = set([start_pos])
        
        while queue:
            current_pos, path = queue.popleft()
            
            for neighbor in self.get_neighbors(current_pos[0], current_pos[1]):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    
                    if neighbor == target_pos:
                        return new_path[1] if len(new_path) > 1 else start_pos
                    
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return start_pos
    
    def get_visible_cells(self, pos, vision_range):
        """Get all cells visible from a position within vision range"""
        visible = set()
        x, y = pos
        
        for dy in range(-vision_range, vision_range + 1):
            for dx in range(-vision_range, vision_range + 1):
                new_x, new_y = x + dx, y + dy
                
                # Check if within bounds
                if (0 <= new_x < self.width and 0 <= new_y < self.height):
                    # Check if within vision radius (circular vision)
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= vision_range:
                        visible.add((new_x, new_y))
        
        return visible
    
    def can_see_target(self, observer_pos, target_pos, vision_range):
        """Check if observer can see target within vision range"""
        distance = math.sqrt(
            (observer_pos[0] - target_pos[0])**2 + 
            (observer_pos[1] - target_pos[1])**2
        )
        return distance <= vision_range
    
    def update_detection_status(self):
        """Update AI detection of player"""
        if self.can_see_target(self.ai_pos, self.player_pos, self.ai_vision_range):
            self.player_detected = True
            self.last_known_player_pos = self.player_pos
            self.turns_since_player_seen = 0
            self.detection_cooldown = 3  # Remember for 3 turns
        else:
            self.turns_since_player_seen += 1
            if self.detection_cooldown > 0:
                self.detection_cooldown -= 1
                self.player_detected = True
            else:
                self.player_detected = False
    
    def generate_patrol_points(self):
        """Generate strategic patrol points"""
        patrol_points = []
        
        # Add points near exits
        for exit_pos in self.exit_positions:
            # Add points around each exit
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (-1, -1)]:
                x, y = exit_pos[0] + dx, exit_pos[1] + dy
                if self.is_valid_position(x, y):
                    patrol_points.append((x, y))
        
        # Add central points
        center_x, center_y = self.width // 2, self.height // 2
        for dx, dy in [(0, 0), (2, 2), (-2, -2), (2, -2), (-2, 2)]:
            x, y = center_x + dx, center_y + dy
            if self.is_valid_position(x, y):
                patrol_points.append((x, y))
        
        # Remove duplicates and invalid points
        valid_patrol_points = []
        for point in patrol_points:
            if point not in valid_patrol_points and point != self.ai_pos:
                valid_patrol_points.append(point)
        
        return valid_patrol_points[:6]  # Limit to 6 patrol points
    
    def get_ai_state_and_target(self):
        """Determine AI state and target based on current situation"""
        if self.player_detected and self.detection_cooldown > 0:
            # Chase mode - player is detected
            return 'chase', self.last_known_player_pos
        elif self.last_known_player_pos and self.turns_since_player_seen < 5:
            # Investigate mode - check last known position
            return 'investigate', self.last_known_player_pos
        else:
            # Patrol mode - guard strategic points
            if not self.patrol_points:
                self.patrol_points = self.generate_patrol_points()
            
            if self.patrol_points:
                target = self.patrol_points[self.current_patrol_target]
                
                # If reached current patrol point, move to next
                if self.ai_pos == target:
                    self.current_patrol_target = (self.current_patrol_target + 1) % len(self.patrol_points)
                    target = self.patrol_points[self.current_patrol_target]
                
                return 'patrol', target
            else:
                # Fallback to guarding nearest exit
                nearest_exit = min(self.exit_positions, 
                                 key=lambda exit: self.bfs_distance(self.ai_pos, exit))
                return 'patrol', nearest_exit
    
    def get_smart_intercept_position(self):
        """Calculate optimal intercept position using advanced BFS analysis"""
        # Find shortest paths from player to each exit
        exit_paths = {}
        for exit_pos in self.exit_positions:
            distance = self.bfs_distance(self.player_pos, exit_pos)
            if distance != float('inf'):
                exit_paths[exit_pos] = distance
        
        if not exit_paths:
            return self.player_pos
        
        # Find the most likely exit (shortest path)
        target_exit = min(exit_paths.keys(), key=lambda x: exit_paths[x])
        
        # Calculate intercept positions along the path
        intercept_candidates = []
        
        # BFS to find all positions along shortest path to target exit
        queue = deque([(self.player_pos, [self.player_pos])])
        visited = set([self.player_pos])
        
        while queue:
            current_pos, path = queue.popleft()
            
            if len(path) > exit_paths[target_exit] + 2:  # Don't go too far
                continue
            
            for neighbor in self.get_neighbors(current_pos[0], current_pos[1]):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    
                    if neighbor == target_exit:
                        # Found path to exit, evaluate intercept positions
                        for i, pos in enumerate(new_path[1:], 1):  # Skip starting position
                            ai_distance = self.bfs_distance(self.ai_pos, pos)
                            player_turns_to_reach = i
                            
                            # AI can intercept if it can reach position before or same time as player
                            if ai_distance <= player_turns_to_reach + 1:
                                intercept_candidates.append((pos, ai_distance, player_turns_to_reach))
                    else:
                        visited.add(neighbor)
                        queue.append((neighbor, new_path))
        
        if intercept_candidates:
            # Choose best intercept position (closest to AI, but effective)
            best_intercept = min(intercept_candidates, 
                               key=lambda x: (x[1], -x[2]))  # Minimize AI distance, maximize player time
            return best_intercept[0]
        
        # Fallback to direct chase
        return self.player_pos
    
    def avoid_getting_stuck(self):
        """Improved stuck prevention using path history analysis"""
        if len(self.ai_path_history) < 3:
            return None
        
        # Check if AI is oscillating between positions
        recent_positions = list(self.ai_path_history)
        if len(set(recent_positions)) <= 2:  # Only 1-2 unique positions in history
            # AI is stuck, find alternative path
            neighbors = self.get_neighbors(self.ai_pos[0], self.ai_pos[1])
            
            # Filter out recently visited positions
            fresh_neighbors = [pos for pos in neighbors if pos not in recent_positions]
            
            if fresh_neighbors:
                return random.choice(fresh_neighbors)
            elif neighbors:
                # All neighbors recently visited, pick one furthest from recent positions
                def distance_from_history(pos):
                    return sum(self.manhattan_distance(pos, hist_pos) for hist_pos in recent_positions)
                
                return max(neighbors, key=distance_from_history)
        
        return None
    
    def get_ai_next_move(self):
        """Enhanced AI decision making"""
        # Update detection status
        self.update_detection_status()
        
        # Check for stuck condition first
        unstuck_move = self.avoid_getting_stuck()
        if unstuck_move:
            return unstuck_move
        
        # Determine AI state and target
        self.ai_state, target = self.get_ai_state_and_target()
        
        # Choose movement strategy based on state and difficulty
        if self.ai_state == 'chase':
            if self.difficulty in ['hard', 'nightmare']:
                # Use smart intercept for higher difficulties
                target = self.get_smart_intercept_position()
            return self.bfs_pathfinding(self.ai_pos, target)
        
        elif self.ai_state == 'investigate':
            # Move towards last known player position
            next_move = self.bfs_pathfinding(self.ai_pos, target)
            
            # If reached investigation target, switch to patrol
            if self.ai_pos == target:
                self.last_known_player_pos = None
                self.investigation_target = None
            
            return next_move
        
        else:  # patrol
            return self.bfs_pathfinding(self.ai_pos, target)
    
    def initialize_game(self):
        """Initialize the game with enhanced features"""
        max_attempts = 30
        for attempt in range(max_attempts):
            self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
            
            if self.place_exits_and_positions():
                obstacle_count = self.get_obstacle_density()
                if self.create_balanced_obstacles(obstacle_count):
                    if self.verify_game_balance():
                        break
            
            if attempt == max_attempts - 1:
                self.create_simple_balanced_game()
        
        # Initialize AI patrol system
        self.patrol_points = self.generate_patrol_points()
        self.current_patrol_target = 0
        
        # Reset game state
        self.game_over = False
        self.game_won = False
        self.turn = 'player'
        self.move_count = 0
        self.max_moves = self.get_max_moves()
        self.player_detected = False
        self.last_known_player_pos = None
        self.detection_cooldown = 0
        self.turns_since_player_seen = 0
        self.ai_path_history.clear()
    
    def place_exits_and_positions(self):
        """Place multiple exits and balanced positions"""
        # Place exits on edges/corners
        edge_positions = []
        
        # Corners
        corners = [(0, 0), (self.width-1, 0), (0, self.height-1), (self.width-1, self.height-1)]
        edge_positions.extend(corners)
        
        # Edge positions
        for x in range(1, self.width-1):
            edge_positions.extend([(x, 0), (x, self.height-1)])
        for y in range(1, self.height-1):
            edge_positions.extend([(0, y), (self.width-1, y)])
        
        # Select exits ensuring they're spread out
        self.exit_positions = []
        while len(self.exit_positions) < self.num_exits and edge_positions:
            exit_candidate = random.choice(edge_positions)
            
            # Ensure exits are not too close to each other
            if not self.exit_positions or all(
                self.manhattan_distance(exit_candidate, existing_exit) >= 4
                for existing_exit in self.exit_positions
            ):
                self.exit_positions.append(exit_candidate)
            
            edge_positions.remove(exit_candidate)
        
        if len(self.exit_positions) < self.num_exits:
            return False
        
        # Place player and AI positions
        valid_positions = []
        for y in range(self.height):
            for x in range(self.width):
                pos = (x, y)
                if pos not in self.exit_positions:
                    valid_positions.append(pos)
        
        # Try multiple combinations for balanced positioning
        for _ in range(100):
            player_candidate = random.choice(valid_positions)
            ai_candidates = [pos for pos in valid_positions 
                           if pos != player_candidate]
            
            if not ai_candidates:
                continue
                
            ai_candidate = random.choice(ai_candidates)
            
            # Check distances to exits and between AI/player
            min_player_to_exit = min(self.manhattan_distance(player_candidate, exit_pos) 
                                   for exit_pos in self.exit_positions)
            min_ai_to_exit = min(self.manhattan_distance(ai_candidate, exit_pos) 
                               for exit_pos in self.exit_positions)
            ai_to_player = self.manhattan_distance(ai_candidate, player_candidate)
            
            # Balanced positioning criteria
            safe_distance = self.get_safe_distance()
            
            if (ai_to_player >= safe_distance and 
                min_player_to_exit <= min_ai_to_exit + 3 and
                min_ai_to_exit <= min_player_to_exit + 4):
                
                self.player_pos = player_candidate
                self.ai_pos = ai_candidate
                return True
        
        return False
    
    def create_balanced_obstacles(self, count):
        """Create obstacles maintaining balance for multiple exits"""
        placed = 0
        attempts = 0
        max_attempts = count * 15
        
        while placed < count and attempts < max_attempts:
            x, y = random.randint(0, self.width-1), random.randint(0, self.height-1)
            pos = (x, y)
            
            # Don't place on important positions
            if (pos in [self.player_pos, self.ai_pos] or 
                pos in self.exit_positions or 
                self.grid[y][x] == 1):
                attempts += 1
                continue
            
            # Temporarily place obstacle
            self.grid[y][x] = 1
            
            # Verify paths to all exits still exist
            player_can_reach_exit = any(
                self.bfs_distance(self.player_pos, exit_pos) != float('inf')
                for exit_pos in self.exit_positions
            )
            
            ai_can_reach_player = self.bfs_distance(self.ai_pos, self.player_pos) != float('inf')
            
            ai_can_reach_exits = any(
                self.bfs_distance(self.ai_pos, exit_pos) != float('inf')
                for exit_pos in self.exit_positions
            )
            
            if player_can_reach_exit and ai_can_reach_player and ai_can_reach_exits:
                placed += 1
            else:
                self.grid[y][x] = 0  # Remove obstacle
            
            attempts += 1
        
        return True
    
    def verify_game_balance(self):
        """Verify game balance with multiple exits"""
        # Check that player can reach at least one exit
        player_can_escape = any(
            self.bfs_distance(self.player_pos, exit_pos) != float('inf')
            for exit_pos in self.exit_positions
        )
        
        # Check that AI can reach player
        ai_can_reach_player = self.bfs_distance(self.ai_pos, self.player_pos) != float('inf')
        
        return player_can_escape and ai_can_reach_player
    
    def create_simple_balanced_game(self):
        """Fallback: create simple balanced game with multiple exits"""
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Simple exit placement
        if self.num_exits >= 3:
            self.exit_positions = [(0, 0), (self.width-1, 0), (self.width-1, self.height-1)]
        elif self.num_exits == 2:
            self.exit_positions = [(0, 0), (self.width-1, self.height-1)]
        else:
            self.exit_positions = [(self.width-1, self.height-1)]
        
        # Simple positioning
        self.player_pos = (self.width//4, self.height//4)
        self.ai_pos = (3*self.width//4, 3*self.height//4)
        
        # Add minimal obstacles
        obstacles = min(8, self.get_obstacle_density() // 4)
        for _ in range(obstacles):
            x, y = random.randint(2, self.width-3), random.randint(2, self.height-3)
            if ((x, y) not in [self.player_pos, self.ai_pos] and 
                (x, y) not in self.exit_positions):
                self.grid[y][x] = 1
    
    def is_valid_position(self, x, y):
        """Check if a position is valid"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid[y][x] == 0
    
    def get_neighbors(self, x, y):
        """Get valid neighboring positions"""
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))
        
        return neighbors
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def check_capture(self):
        """Check if AI has captured the player"""
        return self.ai_pos == self.player_pos
    
    def move_player(self, direction):
        """Move player in specified direction"""
        if self.game_over or self.turn != 'player':
            return False
        
        x, y = self.player_pos
        
        if direction == 'up':
            new_pos = (x, y - 1)
        elif direction == 'down':
            new_pos = (x, y + 1)
        elif direction == 'left':
            new_pos = (x - 1, y)
        elif direction == 'right':
            new_pos = (x + 1, y)
        else:
            return False
        
        # Check if move is valid
        if self.is_valid_position(new_pos[0], new_pos[1]):
            self.player_pos = new_pos
            self.move_count += 1
            
            # Check for immediate capture
            if self.check_capture():
                self.game_over = True
                self.game_won = False
                return True
            
            # Check win condition (any exit)
            if self.player_pos in self.exit_positions:
                self.game_won = True
                self.game_over = True
                return True
            
            # Check move limit
            if self.move_count >= self.max_moves:
                self.game_over = True
                self.game_won = False
                return True
            
            # Switch to AI turn
            self.turn = 'ai'
            return True
        
        return False
    
    def move_ai(self):
        """Move AI with enhanced intelligence"""
        if self.game_over or self.turn != 'ai':
            return
        
        # Record current position in history
        self.ai_path_history.append(self.ai_pos)
        
        # Get next AI move
        next_pos = self.get_ai_next_move()
        
        if next_pos != self.ai_pos:
            self.ai_pos = next_pos
            
            # Check for capture
            if self.check_capture():
                self.game_over = True
                self.game_won = False
                return
        
        # Switch turn back to player
        self.turn = 'player'
    
    def get_player_visible_grid(self):
        """Get grid visible to player based on vision system"""
        visible_cells = self.get_visible_cells(self.player_pos, self.player_vision_range)
        
        # Create visibility grid
        visible_grid = [[-1 for _ in range(self.width)] for _ in range(self.height)]  # -1 = not visible
        
        for x, y in visible_cells:
            visible_grid[y][x] = self.grid[y][x]  # Copy actual grid value
        
        return visible_grid
    
    def get_game_state(self):
        """Get current game state for client"""
        return {
            'grid': self.grid,
            'visible_grid': self.get_player_visible_grid(),
            'player_pos': self.player_pos,
            'ai_pos': self.ai_pos,
            'exit_positions': self.exit_positions,
            'ai_visible': self.can_see_target(self.player_pos, self.ai_pos, self.player_vision_range),
            'player_detected': self.player_detected,
            'ai_state': self.ai_state,
            'game_over': self.game_over,
            'game_won': self.game_won,
            'turn': self.turn,
            'width': self.width,
            'height': self.height,
            'difficulty': self.difficulty,
            'move_count': self.move_count,
            'max_moves': self.max_moves,
            'moves_remaining': self.max_moves - self.move_count,
            'player_vision_range': self.player_vision_range,
            'ai_vision_range': self.ai_vision_range,
            'min_distance_to_exit': min(self.bfs_distance(self.player_pos, exit_pos) 
                                      for exit_pos in self.exit_positions) if self.exit_positions else float('inf'),
            'ai_to_player_distance': self.bfs_distance(self.ai_pos, self.player_pos)
        }

# Global game instance
game = ChamberChaseGame(difficulty='medium')

@app.route('/')
def index():
    return render_template('game.html')

@app.route('/api/game_state')
def get_game_state():
    return jsonify(game.get_game_state())

@app.route('/api/move', methods=['POST'])
def move():
    data = request.get_json()
    direction = data.get('direction')
    
    if game.move_player(direction):
        # After player moves, AI moves automatically
        if not game.game_over and game.turn == 'ai':
            game.move_ai()
    
    return jsonify(game.get_game_state())

@app.route('/api/new_game', methods=['POST'])
def new_game():
    data = request.get_json()
    difficulty = data.get('difficulty', 'medium')
    width = data.get('width', 12)
    height = data.get('height', 12)
    
    global game
    game = ChamberChaseGame(width=width, height=height, difficulty=difficulty)
    return jsonify(game.get_game_state())

if __name__ == '__main__':
    app.run(debug=True)