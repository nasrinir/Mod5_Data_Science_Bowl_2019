from math import pi
import matplotlib.pyplot as plt

def plot_count(feature, title,xlabel, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,6))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')
    g.set_title("Distribution of {}".format(title), fontsize = 16)
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")
    #plt.xlabel(xlabel, fontsize = 16)
    plt.xlabel('')
    plt.ylabel('Count', fontsize = 16)
    plt.xticks(fontsize = 16)
    plt.savefig(f'img/{title}.png', dpi = 500, bbox_inches = 'tight')
    plt.show()
 
def make_spider( df, row, title, color):
    # initialize the figure
    plt.figure(figsize=(10, 10))
    # number of variable
    categories=list(df)
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]

    # Initialise the spider plot
    ax = plt.subplot(1,1,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories, color='k', size=15)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.05,.1,.15], ["5","10","15"], color="grey", size=12)
    plt.ylim(0,0.2)
#     plt.yticks([0.1,.2,.3, 0.4, 0.5], ["10","20","30","40", "50"], color="grey", size=12)
#     plt.ylim(0,0.6)

    # Ind1
    values=df.loc[row].values.flatten().tolist()
    values += values[:1]
    angles += angles[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    #plt.title(title, size=16, color=color, y=1.1)
