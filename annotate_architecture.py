import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def annotate_image():
    # Load the base 1D architecture image
    img = mpimg.imread('reports/figures/fig6_architecture.png')
    
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')
    
    # Add a large, visible text box to explicitly mark it as the Multiclass architecture
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.5, 0.05, "MULTI-HEARTBREAKER ARCHITECTURE\n(Dense 5 Sigmoid Output Layer)", 
            transform=ax.transAxes, fontsize=16, fontweight='bold', 
            verticalalignment='bottom', horizontalalignment='center', bbox=props, color='darkred')
            
    # Save the modified image
    output_path = 'reports/figures/multiclass_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    annotate_image()
