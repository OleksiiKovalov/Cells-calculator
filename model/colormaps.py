"""Defines colormaps used for visualization purposes."""




# Example usage
hex_colors = colormap_to_hex('virdis')
print(hex_colors)
for i, c in enumerate(hex_colors):
    plt.plot([0, 1], [i, i], color=c)
plt.show()
