"""
Script pentru generarea diagramei State Machine a sistemului Text-to-Blender.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_state_machine_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 11.3, 'STATE MACHINE - Text to Blender AI', 
            fontsize=18, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#E8E8E8', edgecolor='black', linewidth=2))
    
    # Define states with positions (x, y, color, width)
    # Flow: IDLE -> RECEIVE -> VALIDATE -> CLASSIFY -> EXTRACT -> GENERATE -> DISPLAY
    #                              |                                    |
    #                              v                                    v
    #                          ERROR <---------------------------------+
    states = {
        'IDLE':            (2, 8, '#90EE90', 2.0),      # Light green - Start
        'RECEIVE\nINPUT':  (5.5, 8, '#87CEEB', 2.2),    # Sky blue
        'VALIDATE\nINPUT': (9, 8, '#87CEEB', 2.2),      # Sky blue
        'CLASSIFY\nINTENT\n(RN)': (12.5, 8, '#FFD700', 2.4),  # Gold - Neural Network
        'EXTRACT\nPARAMS': (12.5, 5, '#87CEEB', 2.2),   # Sky blue
        'GENERATE\nCODE':  (9, 5, '#87CEEB', 2.2),      # Sky blue
        'DISPLAY\nOUTPUT': (5.5, 5, '#90EE90', 2.2),    # Light green - Success
        'ERROR\nHANDLER':  (5.5, 2, '#FF6B6B', 2.2),    # Red - Error
    }
    
    # Draw states
    for state, (x, y, color, width) in states.items():
        box = FancyBboxPatch((x - width/2, y - 0.6), width, 1.2, 
                              boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, state, fontsize=9, fontweight='bold', 
                ha='center', va='center')
    
    # Arrow helper function
    def draw_arrow(start, end, label='', color='black', curve=0, label_offset=(0, 0.3)):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2,
                                   connectionstyle=f"arc3,rad={curve}"))
        if label:
            mid_x = (start[0] + end[0]) / 2 + label_offset[0]
            mid_y = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, fontsize=8, ha='center', va='center',
                   color=color if color != 'black' else '#333333',
                   fontweight='bold')
    
    # Main flow arrows (left to right, then down, then left)
    # IDLE -> RECEIVE_INPUT
    draw_arrow((3, 8), (4.4, 8), 'user input')
    
    # RECEIVE_INPUT -> VALIDATE_INPUT
    draw_arrow((6.6, 8), (7.9, 8), '')
    
    # VALIDATE_INPUT -> CLASSIFY_INTENT (valid path)
    draw_arrow((10.1, 8), (11.3, 8), 'valid', color='#228B22', label_offset=(0, 0.35))
    
    # CLASSIFY_INTENT -> EXTRACT_PARAMS (down)
    draw_arrow((12.5, 7.4), (12.5, 5.6), 'classified', label_offset=(0.7, 0))
    
    # EXTRACT_PARAMS -> GENERATE_CODE (left)
    draw_arrow((11.4, 5), (10.1, 5), 'params')
    
    # GENERATE_CODE -> DISPLAY_OUTPUT (left) - success path
    draw_arrow((7.9, 5), (6.6, 5), 'success', color='#228B22', label_offset=(0, 0.35))
    
    # DISPLAY_OUTPUT -> IDLE (back to start) - curved arrow up and left
    draw_arrow((4.4, 5.3), (2.5, 7.4), 'done\n(new request)', color='#228B22', curve=0.3, label_offset=(-0.5, 0))
    
    # Error paths (red arrows)
    # VALIDATE_INPUT -> ERROR (invalid input)
    draw_arrow((9, 7.4), (6.6, 2.4), 'invalid', color='#DC143C', curve=-0.2, label_offset=(-0.5, 0))
    
    # GENERATE_CODE -> ERROR (template not found)
    draw_arrow((8.5, 4.4), (6.6, 2.5), 'error', color='#DC143C', curve=0.2, label_offset=(-0.3, 0))
    
    # ERROR -> IDLE (retry/reset) - curved arrow
    draw_arrow((4.4, 2.3), (2, 7.4), 'retry', color='#FF8C00', curve=0.4, label_offset=(-0.3, 0))
    
    # Legend box
    legend_x, legend_y = 1, 0.3
    legend_box = FancyBboxPatch((legend_x - 0.3, legend_y - 0.3), 14.6, 1.4,
                                 boxstyle="round,pad=0.05", facecolor='white',
                                 edgecolor='gray', linewidth=1)
    ax.add_patch(legend_box)
    ax.text(8, legend_y + 0.9, 'LEGENDĂ', fontsize=10, fontweight='bold', ha='center')
    
    legend_items = [
        ('#90EE90', 'Start/End'),
        ('#87CEEB', 'Processing'),
        ('#FFD700', 'Neural Network'),
        ('#FF6B6B', 'Error Handler'),
        ('#228B22', '→ Success Path'),
        ('#DC143C', '→ Error Path'),
    ]
    
    for i, (color, label) in enumerate(legend_items):
        x_pos = legend_x + i * 2.4
        if '→' in label:
            ax.plot([x_pos, x_pos + 0.4], [legend_y + 0.3, legend_y + 0.3], 
                   color=color, lw=3)
        else:
            rect = plt.Rectangle((x_pos, legend_y + 0.1), 0.4, 0.4, 
                                 facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
        ax.text(x_pos + 0.5, legend_y + 0.3, label, fontsize=8, va='center')
    
    # Save
    plt.tight_layout()
    plt.savefig('e:/github/Proiect_RN/docs/state_machine.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig('e:/github/Proiect_RN/docs/screenshots/state_machine.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Diagrama salvată în:")
    print("   - docs/state_machine.png")
    print("   - docs/screenshots/state_machine.png")

if __name__ == "__main__":
    create_state_machine_diagram()
