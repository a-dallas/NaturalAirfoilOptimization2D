import matplotlib.pyplot as pp



def generateOGrid(inputFile, outputFile):
    jRange = 55
    initialSpacing = 0.001 #0.000003996 #good for y+ = 1 at Mach = 0.8, 10,000 m. Corresponding Re is 6748182  
    growthParameter = 1.2
    smoothingPasses = 20
    xCoords = []
    yCoords = []
    
    f = open(inputFile, "r")
    lines = f.readlines()
    
    xRow = []
    yRow = []
    for line in lines:
        x , y = line.split(" ")
        x = float(x)
        y = float(y)
        xRow.append(x)
        yRow.append(y)
    
    xCoords.append(xRow)
    yCoords.append(yRow)
    
    for j in range(1, jRange):
        xRow = []
        yRow = []
        
        xVec = [0.0 for i in range(len(xCoords[0]))]
        yVec = [0.0 for i in range(len(xCoords[0]))]
        for i in range(len(xCoords[0])):
            if i == 0 or i+1 == len(xCoords[0]):
                xVec[i] = 1.0
                yVec[i] = 0.0
            else:
                # print(i)
                # print(j)
                # print(yCoords)
                # print(yCoords[0])
                # print(yCoords[0][0])
                xVec[i] = (yCoords[j-1][i-1] - yCoords[j-1][i+1])
                yVec[i] = -(xCoords[j-1][i-1] - xCoords[j-1][i+1])
                mag = (xVec[i]**2.0 + yVec[i]**2.0)**0.5
                xVec[i] = xVec[i]/mag
                yVec[i] = yVec[i]/mag
        
        for k in range(smoothingPasses):
            for i in range(1,int(len(xCoords[0])/5.0)):
                xVec[i] = xVec[i-1]*.25 + xVec[i]*.5 + xVec[i+1]*.25 
                yVec[i] = yVec[i-1]*.25 + yVec[i]*.5 + yVec[i+1]*.25 
                mag = (xVec[i]**2.0 + yVec[i]**2.0)**0.5
                xVec[i] = xVec[i]/mag
                yVec[i] = yVec[i]/mag 
            for i in range(int(len(xCoords[0])*4.0/5.0), len(xCoords[0])-1):
                xVec[i] = xVec[i-1]*.25 + xVec[i]*.5 + xVec[i+1]*.25 
                yVec[i] = yVec[i-1]*.25 + yVec[i]*.5 + yVec[i+1]*.25 
                mag = (xVec[i]**2.0 + yVec[i]**2.0)**0.5
                xVec[i] = xVec[i]/mag
                yVec[i] = yVec[i]/mag
            if j > 5:
                for i in range(int(len(xCoords[0])/5.0), int(len(xCoords[0])*4.0/5.0)):
                    xVec[i] = xVec[i-1]*.4 + xVec[i]*.1 + xVec[i+1]*.4 
                    yVec[i] = yVec[i-1]*.4 + yVec[i]*.1 + yVec[i+1]*.4 
                    mag = (xVec[i]**2.0 + yVec[i]**2.0)**0.5
                    xVec[i] = xVec[i]/mag
                    yVec[i] = yVec[i]/mag
                
        
        for i in range(len(xCoords[0])):
            xRow.append(xCoords[j-1][i]+xVec[i]*initialSpacing)
            yRow.append(yCoords[j-1][i]+yVec[i]*initialSpacing)
        
        initialSpacing = initialSpacing * growthParameter
        xCoords.append(xRow)
        yCoords.append(yRow)
        
    # for k in range(smoothingPasses):
    #     for i in range(1, int(len(xCoords[0])/4)):
    #         for j in range(1,len(xCoords)-1):
    #             xCoords[j][i] = xCoords[j][i-1]*0.01 + xCoords[j][i+1]*0.01 + xCoords[j-1][i]*0.025 + xCoords[j+1][i]*0.025 + xCoords[j][i]*.9
    #             yCoords[j][i] = yCoords[j][i-1]*0.025 + yCoords[j][i+1]*0.025 + yCoords[j-1][i]*0.025 + yCoords[j+1][i]*0.025 + yCoords[j][i]*.9
    
    
    # print(xCoords)
        
    # pp.scatter(xCoords, yCoords, color='blue')
    # pp.scatter(xCoords[0], yCoords[0], color='red')
    # pp.axes().set_aspect('equal')
    # pp.show()
        
        
    g = open(outputFile, 'w')
    g.write("2 \n")
    g.write(str(len(xCoords[0])+1) + " " + str(len(xCoords)) + "\n")
    for j in range(len(xCoords)):
        for i in range(len(xCoords[0])):
            g.write(str(xCoords[j][i]) + "\n")
        g.write(str(xCoords[j][0]) + "\n")
            
    for j in range(len(xCoords)):
        for i in range(len(xCoords[0])):
            g.write(str(yCoords[j][i]) + "\n")
        g.write(str(yCoords[j][0]) + "\n")
            
    g.close()
            

# generateOGrid("customAirfoil.txt", "b")






