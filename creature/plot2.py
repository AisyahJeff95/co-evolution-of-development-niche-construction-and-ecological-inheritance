from matplotlib import pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
from pylab import rcParams
import matplotlib.patches as patches


def trajectory(pop_radius,genomes):
  #rcParams['figure.figsize'] = 15,15

  # c = patches.Circle(xy=(0, 0), radius=pop_radius, ec='black', fill=False)
  x1 = patches.Rectangle((-300,0), width=600, height=2, angle=0.0, color='indigo', fill='indigo') #target
  x2 = patches.Rectangle((-100,12), width=200, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley1
  x3 = patches.Rectangle((-100,37), width=200, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley2
  # x4 = patches.Rectangle((-50,75), width=100, height=10, angle=0.0, color='wheat', fill='wheat')#valley3
  x5, y5 = [-100,100],[50, 50]
  ax = plt.axes()
  # ax.add_patch(c)
  ax.add_patch(x1)
  ax.add_patch(x2)
  ax.add_patch(x3)
  # ax.add_patch(x4)
  ax.plot(x5,y5, color='black')
  for n,i in enumerate(genomes):
    plt.scatter([j[0] for m,j in enumerate(i[1].position) if np.mod(m,10)==0],[j[1] for m,j in enumerate(i[1].position) if np.mod(m,10)==0], s=2.0, color=cm.rainbow(float(n) / len(genomes))) 
    # plt.xlim(50.0, -50.0) # (3)x軸の表示範囲
    # plt.ylim(-50.0, 50.0) # (4)y軸の表示範囲
    plt.xlabel('x direction') # x軸
    plt.ylabel('y direction') # y軸
    ax.set_xlim(-100,100)
    ax.set_ylim(0,pop_radius)
    ax.set_aspect('equal')
    # figureの保存
  plt.savefig(os.path.dirname(__file__) + "/trajectory.png")

def nicheBoard(pop_radius, niche_board):
  
  # Define a colormap
  colormap = plt.cm.get_cmap('cool', len(niche_board))

  fig_board, ax_board = plt.subplots()

  x1 = patches.Rectangle((-200,0), width=400, height=2, angle=0.0, color='indigo', fill='indigo') #target
  x2 = patches.Rectangle((-200,12.5), width=400, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley1
  x3 = patches.Rectangle((-200,37.5), width=400, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley2
  x5, y5 = [-200,200],[50, 50]
  ax_board.add_patch(x1)
  ax_board.add_patch(x2)
  ax_board.add_patch(x3)
  ax_board.plot(x5,y5, color='black')

  for i, board_positions in enumerate(niche_board):
      # Assuming that board_positions is a list of tuples
      x = [coord[0] for coord in board_positions]
      y = [coord[1] for coord in board_positions]

      color = colormap(i / len(niche_board))
      
      ax_board.scatter(x, y, label=f'Board - Generation {i}', c=[color], cmap=colormap, marker='^', s=25)

  ax_board.set_xlabel('X-axis')
  ax_board.set_ylabel('Y-axis')
  sm_board = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=30000))
  sm_board.set_array([])  # An empty array is sufficient
  cbar_board = plt.colorbar(sm_board, ax=ax_board, label='Timesteps')

  ax_board.set_xlim(-150, 150)
  ax_board.set_ylim(0, pop_radius)
  ax_board.set_title('Board Positions')

  fig_board.savefig('board_pos.png', format='png', dpi=300)

def nicheBoardAll(pop_radius, NicheBoard):

    # Define a colormap
    colormap = cm.get_cmap('cool', len(NicheBoard))
  
    fig, ax = plt.subplots()

    x1 = patches.Rectangle((-200,0), width=400, height=2, angle=0.0, color='indigo', fill='indigo') #target
    x2 = patches.Rectangle((-200,12.5), width=400, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley1
    x3 = patches.Rectangle((-200,37.5), width=400, height=3.5, angle=0.0, color='wheat', fill='wheat')#valley2
    x5, y5 = [-200,200],[50, 50]
    ax.add_patch(x1)
    ax.add_patch(x2)
    ax.add_patch(x3)
    ax.plot(x5,y5, color='black')

    # Plot board positions for each generation with color gradients
    for i, board_positions in enumerate(NicheBoard):
        x = [coord[0] for coord in board_positions]
        y = [coord[1] for coord in board_positions]

        color = colormap(i / len(NicheBoard))

        ax.scatter(x, y, c=[color], cmap=colormap, marker='o', s=50, alpha=0.5)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    sm = cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=len(NicheBoard)))
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax, label='Generation')

    ax.set_xlim(-100, 100)
    ax.set_ylim(0, pop_radius)
    ax.set_title('Board Positions over generations')

    fig.savefig('board_positions.png', format='png', dpi=300)
    plt.show()

def partsNum(Genomes,mode=None):
  total_fitness=[]
  parts={}
  best=[]
  for n,genome in enumerate(Genomes):
      b=0
      gen_fitness=[]
      parts[n]=[]
      for ind,g in genome:
          parts[n].append((ind,len(g.substrate.output_coordinates)+1))
          gen_fitness.append(g.fitness)
          if b<g.fitness:
              b=g.fitness
              b_ind=ind
      best.append(b_ind)
      gen_fitness.sort()
      total_fitness.append(gen_fitness)
  
  Ave_parts=[]
  Best_parts=[]
  Most_parts=[]
  for ind,p in parts.items():
    ave_parts=[]
    best_parts=[]
    most_parts=[]
    ave_parts=np.mean(([q[1] for q in p]))
    best_parts=[q[1] for q in p if q[0]==best[ind]]
    most_parts=sorted([(q[1],q[0]) for q in p])[-1]
    Ave_parts.append(ave_parts)
    Best_parts.append(*best_parts)
    Most_parts.append(most_parts)

  plt.figure()
  if mode=="most":
    plt.plot([gen for gen in range(len(Genomes))],Ave_parts,[gen for gen in range(len(Genomes))],Best_parts,[gen for gen in range(len(Genomes))],[M[0] for M in Most_parts])
    plt.legend(["Average","Best","Most"])
  else:
    plt.plot([gen for gen in range(len(Genomes))],Ave_parts,[gen for gen in range(len(Genomes))],Best_parts)
    plt.legend(["Average","Best"])
  plt.xlabel("Generation")
  plt.ylabel("Block num")
  plt.ylim(0,20)
  plt.savefig(os.path.dirname(__file__) + "/partsNum.svg")

  return Most_parts

def fitness(Genomes):
  total_fitness=[]
  for genome in Genomes:
      gen_fitness=[]
      for ind,g in genome:
          gen_fitness.append(g.fitness)

      gen_fitness.sort()
      total_fitness.append(gen_fitness)

  plt.figure()
  plt.plot(np.arange(len(Genomes)),[np.mean(i) for i in total_fitness],np.arange(len(Genomes)),[i[-1] for i in total_fitness])
  plt.ylim(0,60)
  plt.legend(["Average","Best"])
  plt.xlabel("Generation")
  plt.ylabel("Fitness")


def event(pop_radius, Event, Genomes):
    fig, ax = plt.subplots()  # Create a figure and an axes to enable the addition of patches

    # Loop through each possible event position (1 through 4 as per the original code)
    for position in range(1, 5):  # Assuming there could be up to 4 types of events
        mean_values = []
        for generation in Event:
            # Check if the current event position exists in this generation
            if len(generation) >= position:
                # If it exists, calculate the mean value for this event position
                mean_values.append(sum(generation[position-1]) / len(generation[position-1]))
            else:
                # If it doesn't exist, append None to mean_values to skip this generation for the current event
                mean_values.append(None)
        
        if any(value is not None for value in mean_values):
            # Plot the mean values using lines only, without markers
            ax.plot(range(1, len(Event) + 1), mean_values, label=f'Event {position}')
    
    # Add rectangles to the plot
    x2 = patches.Rectangle((len(Genomes), 12), width=len(Genomes), height=3.5, angle=0.0, color='wheat', fill=True)  # valley1
    x3 = patches.Rectangle((len(Genomes), 37), width=len(Genomes), height=3.5, angle=0.0, color='wheat', fill=True)  # valley2
    ax.add_patch(x2)
    ax.add_patch(x3)

    # Plot settings
    plt.title("Developmental events throughout generations")
    plt.ylim(0, pop_radius)
    plt.xlabel("Generations")
    plt.ylabel("Distance")
    plt.legend()
    fig.savefig(os.path.dirname(__file__) + "/growth_event.svg")


