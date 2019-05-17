import json
import operator

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster as sklc
from cycle_detection import detect_cycles
from scipy.spatial.qhull import ConvexHull
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Show forloop progress

# Pomegranete hidden markov

court_img = mpimg.imread('baseketball_court_flat.jpg')
passCounter = np.Inf
shotCounter = np.Inf
dribbleCounter = np.Inf


# This function formats the data frames a bit nicers and easier to read
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')


# Just opens the data file, converts and prints it into somethhing readable as it's all written on one line
def getEvents():
    with open('0021500001.json', 'r') as f:
        nba_dict = json.load(f)
    nba_data = pd.DataFrame(nba_dict)

    events = pd.DataFrame(nba_data['events'][0])

    # neo_nba = pd.concat({k: pd.DataFrame(v) for k, v in events.items()}, axis=0)
    return events


def getMoments():
    moments = []
    for header_moments in getEvents():
        moments.extend(header_moments.get('moments'))
    gameinfo = pd.DataFrame(moments, columns=["Quarter", "Absolute Time", "Game Time Left", "Shot Clock End?", "Null",
                                              "Position Data"])
    # print(moments[5][5])
    return moments, gameinfo


def getposData():
    posData = []

    for point_in_time in moments:
        for pos in point_in_time[5]:
            posData.append(pos)
    return posData


#  TODO Cluster Possesion time with ball diameter to see if the clusters can group how likely you will take a shot?

# When team and player ID is -1, -1 then it is referring to the ball
# If ball diameter is 0.0000 then it is referring to a player
# Every 11 positions is a new frame as it will update a new ball size and position
# This is about 0.4 seconds

# TODO Consider possession per quarter

previousHandler = np.Inf
count = 0


def update_pos_pp(id, i, team):
    # Update possession time per player
    global id_dict
    global previousHandler
    global count
    global passCounter

    def calcElapsed(time, i):
        global gameStart
        global team_pos
        currentTime = moments[i][2]
        difference = time - currentTime
        gameStart = currentTime
        # print("difference is {}".format(difference))
        # print("currentTime is {}".format(currentTime))
        if difference < 0:
            difference = 1
        return difference

    amount = calcElapsed(gameStart, i)
    id_dict[id] += amount
    team_pos[team] += amount
    if previousHandler != id:
        previousHandler = id
        count += 1
    if count > 2:
        print("PASS WAS MADE")
        passCounter += 1
        count = 0

    # print(id_dict)


moments, gameinfo = getMoments()
gameStart = moments[0][2]
posData = getposData()
df = pd.DataFrame(posData, columns=['teamId', 'playerId', 'x_pos', 'y_pos', 'ball_diameter'])
df = df.mask(df == 0)
playerids = df['playerId'].unique().tolist()
id_dict = {i: 0 for i in playerids}
team_pos = {'a': 0, 'b': 0}

P = np.array(posData)
M = P.shape[0]

mm = M // 11


# TODO Try removing duplicates to see if the data still glitches around

# gameinfo['next_abs_time'] = gameinfo['Absolute Time'].shift(-1)
# mask = gameinfo['Absolute Time'] == gameinfo['next_abs_time']
def detectTimeCycles():
    timestamp = gameinfo['Absolute Time']
    cycles = detect_cycles(timestamp)
    print(cycles)
    print(timestamp[timestamp.duplicated()])
    gameinfo.drop_duplicates(subset=['Absolute Time'], keep=False, inplace=True)
    print(gameinfo.shape[0], df.shape[0])


#


# print(gameinfo.loc[gameinfo['time_glitch']]== True)
fig1, ax = plt.subplots(figsize=(10, 7))


# plt.ion()
def initaliseAnim():
    ax.autoscale(enable=True)
    plotimg = plt.imshow(court_img, aspect='auto', interpolation='none', zorder=0, extent=[0, 94, 50, 0])
    ax.axis('off')
    ax.set(xlim=(0, 94), ylim=(0, 50))
    # defense = ax.scatter([],[])
    # offense = ax.scatter([],[])
    # ball = ax.scatter([],[])
    hulld = []
    hullo = []
    d_fill, = ax.fill(0, 0, fill=True)
    o_fill, = ax.fill(0, 0, fill=True)
    newP = np.array(posData)
    defense = ax.scatter(x=[], y=[], marker='o', c='yellow', s=500)
    offense = ax.scatter(x=[], y=[], marker='o', c='green', s=500)
    ball = ax.scatter(x=[], y=[], marker='o', c='orange', s=100)
    direction = ax.plot(x=[], y=[])
    return defense, offense, ball, direction,


# for i in range(1200,1550):
#     print( np.array([P[11 * i, 2], P[11 * i, 3]]).T)
#
# print(df['ball_diameter'].max())
# print(df['ball_diameter'].min())
# print(df['ball_diameter'].mean())
# print(df['ball_diameter'].mode())
# print(df['ball_diameter'].median())
# print(df['ball_diameter'].std())

ball_df = df[df['teamId'] == df['teamId'].unique().tolist()[0]]

selection = df['teamId'] == df['teamId'].unique().tolist()[
    1]  # Dynamically select the first unique team id  and return list of boolean where it exists
notball = df['teamId'] == df['teamId'].unique().tolist()[0]
criteria = ~selection & ~notball  # Tilda means NOT in pandas
# print(selection)
team1_df = df[selection]
team2_df = df[criteria]
print(team1_df.shape[0], team2_df.shape[0])

active1 = []
active2 = []


# Creates list with true/false values to represent what should be actively drawn

def populateisActive(active, team):
    for p in range(team.shape[0]):
        if p < 5:
            active.append(True)
        else:
            active.append(False)
    return active


active1 = populateisActive(active1, team1_df)
active2 = populateisActive(active2, team2_df)
print(len(active1), len(active2))


def animate(i):
    global active1
    global active2
    # x = ball_df['x_pos'].head(100)
    # y = ball_df['y_pos'].head(100)
    # velX = x.diff()
    # velY = y.diff()
    # print(i)
    # players = newP[11 * i:11 * (i + 1), 1].astype(int)
    # labels = ['{}'.format(j) for j in players]
    points_defense = P[11 * i:11 * i + 5, 2:4]
    points_offense = P[(11 * i) + 5:11 * i + 10, 2:4]
    hulld = ConvexHull(points_defense)
    hullo = ConvexHull(points_offense)
    defense, offense, ball, directions = initaliseAnim()
    # directions.set_ydata(x+velX)
    currentRow1 = team1_df[['x_pos', 'y_pos']][active1]
    currentRow2 = team2_df[['x_pos', 'y_pos']][active2]
    defense.set_offsets(currentRow1.values)
    offense.set_offsets(currentRow2.values)

    # Shift down the active values by 5 every frame for each team mate on court
    active1 = np.roll(active1, 5)
    active2 = np.roll(active2, 5)
    # defense.set_offsets(currentRow1)
    ball.set_offsets(ball_df[['x_pos', 'y_pos']].iloc[i - 1])

    # nbrs = NearestNeighbors(n_neighbors= 2, algorithm= 'ball_tree').fit()
    # distances, indicies = nbrs.kneighbors(df[['x_pos','y_pos']])

    # getCurrentPossession(i,ball_df[['x_pos','y_pos']].iloc[i-1])

    # offense.set_offsets([[P[11 * i + 5:11 * i + 10, 2], P[11 * i + 5:11 * i + 10, 3]]])
    # ball.set_offsets([[P[11 * i, 2], P[11 * i, 3]]])
    return defense, offense, ball


def getCurrentPossession(i, ball_pos):
    def euclideanDistance(x1, x2):
        return np.linalg.norm(x1 - x2)

    def getNeighbours(ball, players):
        distances = []
        neighbour = []
        for index in range(0, len(players)):
            dist = euclideanDistance(players[index], ball)
            distances.append((index, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbour.append(distances[0][0])
        return neighbour[0]

    def getNearestID(team):
        _team = []
        if team == 'a':
            xypos = team1_df[['x_pos', 'y_pos']]
            AplayerId = team1_df['playerId']
        else:
            xypos = team2_df[['x_pos', 'y_pos']]
            AplayerId = team2_df['playerId']
        for x in range(0, 5):
            _team.extend([np.array(xypos[['x_pos', 'y_pos']].iloc[x - 1])])
        return int(AplayerId.iloc[getNeighbours(ball_pos, _team)]), _team[getNeighbours(ball_pos, _team)]

    # Team A is Green team
    teamANearestID, teamADist = getNearestID(team='a')
    teamBNearestID, teamBDist = getNearestID(team='b')

    def hasPossession(distA, distB):
        eudistA = euclideanDistance(distA, ball_pos)
        eudistB = euclideanDistance(distB, ball_pos)
        # print(eudistA,eudistB)
        if eudistA < eudistB:
            update_pos_pp(teamANearestID, i, team='a')
            return "Team A"
        elif eudistB < eudistA:
            update_pos_pp(teamBNearestID, i, team='b')
            return "Team B"
        elif eudistA == eudistB:
            return "50/50 ball"

    # print(teamANearestID,teamBNearestID)
    return hasPossession(teamADist, teamBDist)


#  Turn distances from ball into matri

# TODO Find out the timestamps of the data through absolute time and see if why the data glitches
# Progress update --  found duplicates in absolute time (shouldn't happen as time can't repeat itself)
#  Attempted to remove duplicates however data still repeats itself
# TODO  plot average over time of who has the ball most


# TODO Get some visuaisations

# TODO look at piplining some models

# TODO Run 1NN on ball

#
#
ani = animation.FuncAnimation(fig1, animate, frames=len(P), repeat=True, blit=True)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='me'), bitrate=1800)
# # plt.draw()
plt.show()


# print((11*222826+6)%len(P))
# print(P[(11 * 222826 + 6)%len(P):(11 * 222826 + 11)%len(P), 2])
# Row 222826 will return an empty array when getting x and y pos


def generatePossessionsPerPlayer():
    ball_events = ball_df.shape[0]
    print("Running possession anaylatics")
    print(ball_df.shape[
              0])  # this should be the actual length of loop, however it's very timely, and will only restrict to small loops for testing
    for i in tqdm(range(0, 550)):  # This will time the for loop and show progress
        getCurrentPossession(i, ball_df[['x_pos', 'y_pos']].iloc[i - 1])
        pass
    print("Finished counting possessions per player in minutes")  #
    return id_dict


# plt.figure(2)

# plt.bar(range(len(id_dict)), id_dict.values(), align='center')
# plt.xticks(range(len(id_dict)), list(id_dict.keys()),rotation='vertical')
#
# plt.figure(3)
# plt.bar(range(len(team_pos)), team_pos.values(), align='center')
# plt.xticks(range(len(team_pos)), list(team_pos.keys()),rotation='vertical')

# plt.show()

#  Assuming this will hold a dictionary of time spent in seconds for each player

# %% Get Possessions
id_dict__ = generatePossessionsPerPlayer()
print(id_dict__)
possessions = pd.DataFrame(id_dict__, index=[0]).T
possessions = possessions.rename_axis('-1').rename_axis(None, 1).reset_index()
possessions.columns = ['player_id', 'possession_time']
# possessions['ball_diameter_mean']= df['ball_diameter'].rolling(window=55).mean()
# mode =df[df.ball_diameter>6].mean()
# possessions['mode_poss_time']=possessions.apply(lambda row: (row.possession_time/mode) *100,axis=1)
# possessions['mode_poss_time']= posseions['possession_time']/ mode *100

# print(possessions)
# possessions.assign(ball_mode=df['ball_diameter'].mode())
# %% Get ball diameter means rolling every 5 frames and split train test
rolling_ball_mean = df['ball_diameter'].dropna().rolling(window=5, min_periods=5).mean()
# rolling_ball_mean.fillna(df['ball_diameter'].mean())
rolling_ball_mean = rolling_ball_mean.fillna(6.5)
print(rolling_ball_mean)
# x =possessions.possession_time / possessions.mode_poss_time *100
print("Split test/trainData")
X_train, X_test = train_test_split(rolling_ball_mean.values.reshape(-1, 1), test_size=0.4, shuffle=False)
# .values.reshape(-1,1) is because there is only 1 feature being input where normally 2 is input, requires to be reshaped.
print(X_train)

# Perhaps plot mean ball height divided by possession time for each player
# The more you hold the ball the more likely you are to shoot? KCluster? height, possession in order
# %% Estimate Bandwidth
# height_possession = pd.concat([possessions,ball_height],axis=1)
print("calculating bandwidth")
# bandwidth = sklc.estimate_bandwidth(X_train,n_jobs=-1,n_samples=len(X_train)//4)
print("Creating Model")
# %% Make Meanshift Model
# shotModel= sklc.MeanShift(bandwidth=bandwidth).fit(X_train)

# %% Make KMeans Model
shotModel2 = sklc.KMeans(n_clusters=4, verbose=1, n_init=20).fit(X_train)

# %% Mkae Predictions
print("Prediciting using test data")
# shotLabels1 = shotModel.predict(X_test)
# print(pd.DataFrame(shotLabels1))
print("Attempting KMeans")

shotLabels2 = shotModel2.predict(rolling_ball_mean.values.reshape(-1, 1))
shotLabels2 = pd.DataFrame(shotLabels2)

print(shotLabels2)
# %%
# TODO Count how often the values change between shot, dribble and pass and make HMM
centroids = np.sort(shotModel2.cluster_centers_.flatten())
lowL, midL, highmidL, highL = shotModel2.predict(centroids.reshape(-1, 1))  # put clusters in variables
df['rolling_mean'] = rolling_ball_mean
# df.plot().scatter(x='x_pos',y='y_pos',c='rolling_mean')
# plt.show()
indicies = rolling_ball_mean.index
# pd.DataFrame(shotLabels2,index=indicies)
# shotLabels2= pd.Series(shotLabels2)
df['Labels'] = shotLabels2.reindex(rolling_ball_mean.index)

df['Labels'].value_counts()
maybeShots = df['Labels'] == float(highL)
shotCounter = maybeShots.value_counts()

# %%Get the Trajectory of the ball
from rdp import rdp

ball_df.head(100).plot.scatter(x='x_pos', y='y_pos');
plt.show()
plt.clf()
x = ball_df['x_pos'].head(100)
y = ball_df['y_pos'].head(100)
velX = x.diff()
velY = y.diff()

x = x.values.reshape(1, -1)
y = y.values.reshape(1, -1)
simpleX = rdp(x).flatten()

simpleY = rdp(y).flatten()

plt.plot(simpleX, simpleY)
plt.show()
plt.clf()

x = x.flatten()
y = y.flatten()
velX = velX.values.reshape(1, -1).flatten()
velY = velY.values.reshape(1, -1).flatten()

plt.arrow(47, 27, 2, 2)

#
# x = x.values.reshape(1,-1)
# y = y.values.reshape(1,-1)
# velX = velX.values.reshape(1,-1)
# velY = velY.values.reshape(1,-1)

plt.show()

#
#
# for i in range(mm):
#     players = newP[11 * i:11 * (i + 1), 1].astype(int)
#
#     labels = ['{}'.format(j) for j in players]
#     points_defense = P[11 * i:11 * i + 5, 2:4]
#     print(points_defense)
#     # print(points_defense)
#
#     points_offense = P[(11 * i) + 5:11 * i + 10, 2:4]
#     hulld = ConvexHull(points_defense)
#
#     hullo = ConvexHull(points_offense)
#     # print(points_defense[hulld.vertices,0])
#     # fig1 = convex_hull_plot_2d(ConvexHull(points_defense))
#     # fig1.show()
#     # convex_hull_plot_2d(ConvexHull(points_offense))
#
#     # plt.imshow(figg, zorder=0, aspect= 'auto', extent = [0, 94, 50, 0])
#     plt.fill(points_defense[hulld.vertices, 0], points_defense[hulld.vertices, 1], fill=True,
#              color='yellow', alpha=0.5, edgecolor='yellow')
#     plt.fill(points_offense[hullo.vertices, 0], points_offense[hullo.vertices, 1], fill=True,
#              color='green', alpha=0.5, edgecolor='green')
#
#     plt.scatter(x=P[11 * i + 1:11 * i + 5, 2], y=P[11 * i + 1:11 * i + 5, 3], marker='o', c='yellow', s=500)
#     for label, x, y in zip(range(1, 5), P[11 * i + 1:11 * i + 5, 2], P[11 * i + 1:11 * i + 5, 3]):
#         plt.annotate(str(labels[label]), xy=(x - 0.5, y - 0.5), color='black')
#         plt.scatter(x=P[11 * i + 5:11 * i + 10, 2], y=P[11 * i + 5:11 * i + 10, 3], marker='o', c='green', s=500)
#         for label, x, y in zip(range(5, 11), P[11 * i + 5:11 * i + 10, 2], P[11 * i + 5:11 * i + 10, 3]):
#             plt.annotate(str(labels[label]), xy=(x - 0.5, y - 0.5), color='blue')
#             plt.scatter(x=P[11 * i, 2], y=P[11 * i, 3], marker='o', c='orange', s=100)
#
#             plt.pause(0.00000001)
#
#     plt.clf()


# ani = animation.FuncAnimation(fig1, animate, frames=17, repeat=True)
# plt.show()

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata =dict(artist = 'me'), bitrate=1800)
#
# fig = plt.figure(figsize=(10,6))
# plt.xlim(0, 94)
# plt.ylim(0, 50)
# plt.xlabel('Year',fontsize=20)
#
# def plotData(ndarray):
#     pass


# plt.draw()
# for i in range(mm):
#
#     newP = np.array(posData)
#     players = newP[11*i:11*(i+1),1].astype(int)
#
#     labels = ['{}'.format(j) for j in players]
#     points_defense = P[11 * i :11 * i + 5, 2:4]
#     print(points_defense)
#     # print(points_defense)
#
#     points_offense = P[(11 * i) + 5 :11 * i + 10, 2:4]
#     hulld  = ConvexHull(points_defense)
#
#     hullo = ConvexHull(points_offense)
#     # print(points_defense[hulld.vertices,0])
#     # fig1 = convex_hull_plot_2d(ConvexHull(points_defense))
#     # fig1.show()
#     # convex_hull_plot_2d(ConvexHull(points_offense))
#
#     # plt.imshow(figg, zorder=0, aspect= 'auto', extent = [0, 94, 50, 0])
#     plt.fill(points_defense[hulld.vertices,0], points_defense[hulld.vertices,1], fill=True,
#              color= 'yellow', alpha = 0.5, edgecolor ='yellow')
#     plt.fill(points_offense[hullo.vertices, 0], points_offense[hullo.vertices, 1], fill=True,
#              color='green', alpha = 0.5, edgecolor ='green')
#
#
#
#     plt.scatter(x=P[11 * i+1:11 * i + 5, 2], y=P[11 * i+1:11 * i + 5, 3], marker='o', c ='yellow', s = 500)
#     for label, x, y in zip(range(1,5), P[11 * i+1:11 * i + 5, 2], P[11 * i+1:11 * i + 5, 3]):
#         plt.annotate(str(labels[label]), xy=(x - 0.5, y - 0.5), color='black')
#         plt.scatter(x=P[11 * i + 5:11 * i + 10, 2], y=P[11 * i + 5:11 * i + 10, 3], marker='o', c ='green', s = 500)
#         for label, x, y in zip(range(5, 11), P[11 * i + 5:11 * i + 10, 2], P[11 * i + 5:11 * i + 10, 3]):
#             plt.annotate(str(labels[label]), xy=(x - 0.5, y - 0.5), color='blue')
#             plt.scatter(x=P[11*i, 2], y=P[11 * i, 3], marker='o', c ='orange', s = 100)
#
#             plt.pause(0.00000001)
#
#     plt.clf()
#

# # plt.show()
#
# # Writer = animation.writers['ffmpeg']
# # writer = Writer(fps=20, metadata =dict(artist = 'me'), bitrate=1800)
# #
# # fig = plt.figure(figsize=(10,6))
# # plt.xlim(0, 94)
# # plt.ylim(0, 50)
# # plt.xlabel('Year',fontsize=20)
# #
# # def plotData(ndarray):
#     pass
