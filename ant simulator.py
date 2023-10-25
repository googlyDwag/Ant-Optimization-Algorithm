from enum import Enum
import math
import random
import time
import pygame
import numpy as np  #I'm switching to numpy for vectors for opimization purposes


pygame.font.init()
font = pygame.font.SysFont(None, 36)

antNum = 2
boardSize = np.array([20, 8])
cellSize = np.array([80, 80])
headingWeight = 0.8
pullWeight = 1 - headingWeight
quitChance = 1

def normalize(vector, normTo):
    vector /= np.linalg.norm(vector)
    vector *= normTo
def rotate(vector, radians):
    ogVector = np.copy(vector)
    vector[0] = ogVector[0]*math.cos(radians) - ogVector[1]*math.sin(radians)
    vector[1] = ogVector[0]*math.sin(radians) + ogVector[1]*math.cos(radians)


class Cell:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.pheremones = []
        self.archivePheremones = []
        self.food = []


class Board:
    def __init__(self, size, cellSize, antNum, foodInfo, deltaT, screen):
        self.size = size
        self.cellSize = cellSize
        self.antNum = antNum
        self.foodInfo = foodInfo
        self.deltaT = deltaT

        self.pixelSize = cellSize * size

        self.board = []
        self.ants = []
        self.frozenAnts = []
        self.screen = screen

        for row in range(1, self.size[1] + 1):
            self.board.append([])
            for column in range(1, self.size[0] + 1):
                self.board[row - 1].append(
                    Cell(np.array([column, row]), self.cellSize))

        for food in foodInfo:
            self.board[np.floor(food[1]/self.cellSize[1]).astype(int)][np.floor(food[0]/self.cellSize[0]).astype(int)].food.append(food)

    def print(self):
        print("\n")
        for ant in self.ants:
            print("")
            ant.print()

    def interGeneration(self):
        for row in self.board:
            for cell in row:
                for pheremone in cell.pheremones:
                    pygame.draw.circle(self.screen, (122, 235, 143), (pheremone[0], pheremone[1]), math.sqrt(pheremone[4])/8)

        for food in self.foodInfo:
            pygame.draw.circle(self.screen, (52, 229, 235), (food[0], food[1]), food[2]*0.9+8)

    def step(self):
        deltaT = self.deltaT
        for ant in self.ants:
            if ant.frozen == False:
                ant.varyHeading(deltaT)
                ant.move()

    def calculatePathScore(self, ant):
        foodVisitedNp = np.array(ant.foodsVisited)[1:]
        if foodVisitedNp.size != 0:
            return (np.sum(foodVisitedNp[:, 2])**2 - np.sum(foodVisitedNp[:, 2]**2)/ant.pheremones.shape[0]**5)*30
        else:
            return 0
    def startGeneration(self):
        antID = 0
        for spawn in self.foodInfo:
            for ant in range(self.antNum):
                self.ants.append(Ant(spawn[0:2], self, antID, self.deltaT, spawn))
                antID += 1

    def endGeneration(self):

        for row in self.board:
            for cell in row:
                for pheremoneIndex, pheremone in enumerate(cell.pheremones):
                    pheremone[4] *= 0.8
                    if pheremone[4] < 5 and pheremone[4] > 0:
                        cell.pheremones.pop(pheremoneIndex)

        self.ants = np.concatenate((self.ants, self.frozenAnts))

        for ant in self.ants:
            ant.pheremones = np.array(ant.pheremones)
            ant.pheremones[:, 4] = self.calculatePathScore(ant)
            if ant.pheremones[0][4] != 0:
                for pheremone in ant.pheremones:
                    self.board[pheremone[3].astype(int)][pheremone[2].astype(int)].pheremones.append(pheremone)

        self.ants = []#Clear ants

class Ant:
    def __init__(self, position, board, num, deltaT, foodSpawn):
        self.position = np.copy(position)
        self.board = board
        self.num = num
        self.cellPos = np.floor(self.position /
                                self.board.cellSize).astype(int)
        self.cell = board.board[self.cellPos[1]][self.cellPos[0]]
        self.frozen = False

        self.pheremones = []
        self.foodsVisited = [foodSpawn]

        self.headingRandomness = 0.15
        self.speed = 10
        self.searchDistance = 20
        self.scaledSearch = self.speed*deltaT*self.searchDistance

        if len(self.cell.pheremones) != 0:
            self.heading = self.filterlessPheremone()
        else:
            self.heading = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])

    def foodMovement(self):
        global quitChance
        for food in self.cell.food:
            for visitedFood in self.foodsVisited:
                if np.all(food==visitedFood):
                    return
            pygame.draw.line(self.board.screen, (255, 255, 255), (self.position[0], self.position[1]), (food[0], food[1]), 2)
            self.pheremones.append(np.concatenate((self.position, self.cellPos, np.empty(1))))
            self.position[0] = food[0]
            self.position[1] = food[1]
            self.foodsVisited.append(food)
            if random.random() < quitChance:
                self.frozen = True

    def print(self):
        print("Position:")
        print(self.position)
        print("Heading:")
        print(str(self.heading) + "\n")

    def findPheremonePull(self):
        pulls = []
        intCellPos = self.cellPos.astype(int)
        global boardSize
        
        for row in range(max(intCellPos[1] - 1, 0), min(intCellPos[1] + 2, boardSize[1])):
            for column in range(max(intCellPos[0] - 1, 0), min(intCellPos[0] + 2, boardSize[0])):
                for pheremone in self.board.board[row][column].pheremones:
                    relativePheremonePosition = pheremone[0:2] - self.position
                    dotProduct = np.dot(relativePheremonePosition, self.heading)
                    if dotProduct > 0 and np.linalg.norm(relativePheremonePosition) < 60:
                        pulls.append((relativePheremonePosition)*(pheremone[4]**3))
        pulls.append(self.heading)
        pulls = np.array(pulls) #Freezing goes br

        return np.mean(pulls, axis=0)

    def filterlessPheremone(self):
        pulls = []
        intCellPos = self.cellPos.astype(int)
        global boardSize
        
        for row in range(max(intCellPos[1] - 1, 0), min(intCellPos[1] + 2, boardSize[1])):
            for column in range(max(intCellPos[0] - 1, 0), min(intCellPos[0] + 2, boardSize[0])):
                for pheremone in self.board.board[row][column].pheremones:
                    relativePheremonePosition = pheremone[0:2] - self.position
                    if np.linalg.norm(relativePheremonePosition) < 60:
                        if relativePheremonePosition[0] != 0 or relativePheremonePosition[1] != 0:
                            pulls.append((relativePheremonePosition)*(pheremone[4]**3))
        pulls = np.array(pulls) #Freezing goes br

        return np.mean(pulls, axis=0)

    def varyHeading(self, deltaT):
        global headingWeight
        global pullWeight
        
        pull = self.findPheremonePull()
        #pygame.draw.line(self.board.screen, (0, 255, 0), (self.position[0], self.position[1]), (self.position[0] + pull[0], self.position[1] + pull[1]), 4) #Draws pull
        normalize(pull, self.speed*deltaT)

        self.heading = headingWeight*self.heading + pullWeight*pull

        rotate(self.heading, random.uniform(-deltaT*self.headingRandomness, deltaT*self.headingRandomness))

        normalize(self.heading, self.speed * deltaT)
        nextPosition = np.copy(self.position)
        nextPosition = nextPosition + self.heading

        if nextPosition[0] < 0 or nextPosition[0] > self.board.pixelSize[0]:
            self.heading = self.heading * np.array([-1, 1])
        if nextPosition[1] < 0 or nextPosition[1] > self.board.pixelSize[1]:
            self.heading = self.heading * np.array([1, -1])

    def move(self):
        self.foodMovement()

        self.pheremones.append(np.concatenate((self.position, self.cellPos, np.empty(1))))

        prePos = np.copy(self.position)
        self.position = self.position + self.heading

        pygame.draw.line(self.board.screen, (255, 255, 255), (prePos[0], prePos[1]), (self.position[0], self.position[1]), 2)
        #pygame.draw.line(self.board.screen, (255, 0, 0), (self.position[0], self.position[1]), (self.position[0] + self.heading[0], self.position[1] + self.heading[1]), 4)

        self.cellPos = np.floor(self.position / self.board.cellSize).astype(int)
        self.cell = self.board.board[self.cellPos[1]][self.cellPos[0]]





#Surface with α channel
translucent_surface = pygame.Surface((boardSize[0]*cellSize[0], boardSize[1]*cellSize[1]), pygame.SRCALPHA)
translucent_surface.fill((0, 0, 0, 150))  # The last value sets the transparency level

foodEditing = True

generationLength = 120

screen = pygame.display.set_mode((boardSize[0]*cellSize[0], boardSize[1]*cellSize[1]))
enlargening = False

while True:
    sacrifice = False

    foods = []
    newestFood = [0, 0, 0]
    while foodEditing == True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pos()[0] > boardSize[0]*cellSize[0] - 240 and pygame.mouse.get_pos()[1] < 40:
                    if sacrifice == True:
                        foodEditing = False
                    sacrifice = True
                else:
                    newestFood = [pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1], 0]
                    enlargening = True
            
            if event.type == pygame.MOUSEBUTTONUP:
                foods.append(np.array(newestFood))                      
                enlargening = False

        if newestFood[2] < 10:
                newestFood[2] = newestFood[2] + 0.05

        for food in foods:
            pygame.draw.circle(screen, (52, 229, 235), (food[0], food[1]), food[2]*0.9+8)
        if enlargening == True:
            pygame.draw.circle(screen, (52, 229, 235), (newestFood[0], newestFood[1]), newestFood[2]*0.9+8)

        pygame.draw.polygon(screen, (237, 100, 113), [(boardSize[0]*cellSize[0], 0), (boardSize[0]*cellSize[0] - 240, 0), (boardSize[0]*cellSize[0] - 200, 40), (boardSize[0]*cellSize[0], 40)])
        img = font.render('Start Simulation', False, (255, 255, 255))
        screen.blit(img, (boardSize[0]*cellSize[0] - 200, 10))
        
        img = font.render('Press and hold to create food', False, (255, 255, 255))
        screen.blit(img, (10, 10))

        img = font.render('Size of the food is the value', False, (255, 255, 255))
        screen.blit(img, (10, 40))

        pygame.display.flip()

    foods = np.array(foods)

    print(foods)

    alpha = Board(boardSize, cellSize, 2, foods, 2, screen)

    sacrifice = False

    while foodEditing == False:
        alpha.screen.blit(translucent_surface, (0, 0))
        alpha.interGeneration()

        pygame.draw.polygon(screen, (237, 100, 113), [(boardSize[0]*cellSize[0], 0), (boardSize[0]*cellSize[0] - 240, 0), (boardSize[0]*cellSize[0] - 200, 40), (boardSize[0]*cellSize[0], 40)])
        img = font.render('End Simulation', False, (255, 255, 255))
        screen.blit(img, (boardSize[0]*cellSize[0] - 200, 10))

        alpha.startGeneration()
        for step in range(generationLength):               
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.mouse.get_pos()[0] > boardSize[0]*cellSize[0] - 240 and pygame.mouse.get_pos()[1] < 40:
                        if sacrifice == True:
                            foodEditing = True
                        sacrifice = True

            alpha.step()
            pygame.display.flip()
            pygame.time.Clock().tick(120)
        alpha.endGeneration()
    
    screen.fill((0, 0, 0))










