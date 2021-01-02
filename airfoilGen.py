import jax.numpy as np
from jax import jacfwd
import jax.ops as jo
from math import factorial

#For tuning the spacing of x and u values
initStep = 0.0001
scalingRatio = 1.1
initStep_U = 0.02
scalingRatio_U = 1.5
initStep_UBack = 0.031
scalingRatio_UBack = 1.0
outFile = "customAirfoil.txt"


xVals = [0.0]
backendFlag = 0
while xVals[-1] < 1.0:
    
    newLoc = initStep*scalingRatio + xVals[-1]
    initStep = initStep*scalingRatio
    
    if newLoc > 0.5 and backendFlag == 0:
        backendFlag = 1
        scalingRatio = (1.0/scalingRatio)**0.8
    
    if newLoc > 1.0:
        newLoc = 1.0
    
    xVals.append(newLoc)
  
xVals = xVals[::-1] + xVals[1:]


uVals = [0.0]
while uVals[-1] < 1.0:
    
    newLoc = initStep_U*scalingRatio_U + uVals[-1]
    initStep = initStep_U*scalingRatio_U
    
    if newLoc > 1.0: # don't add this to avoid duplicates
        newLoc = 1.0
        break
    else:
        uVals.append(newLoc)
    

uValsBack = [0.0]
while uValsBack[-1] < 1.0:
    
    newLoc = initStep_UBack*scalingRatio_UBack + uValsBack[-1]
    initStep = initStep_UBack*scalingRatio_UBack
    
    if newLoc > 1.0: # don't add this to avoid duplicates
        newLoc = 1.0
        break
    else:
        uValsBack.append(newLoc)


def fourDigitAirfoil(x):
    yVals = np.array([0.0 for x in range(len(xVals))])
    m = x[0]/100.0
    p = x[1]/10.0
    t = x[2]/100.0
    for i in range(len(yVals)):
        xC = xVals[i]
        y_t = 5*t*(0.2969*xC**0.5 - 0.1260*xC - 0.3516*xC**2 + 0.2843*xC**3 - 0.1015*xC**4)
        y_c = 0.0
        if xC < p:
            y_c = m*(2*p*xC-xC**2)/p**2
        else:
            y_c = m*((1-2*p)+2*p*xC-xC**2)/(1-p)**2
            
        if i < len(yVals)/2:
            yVals = jo.index_update(yVals, i, y_c - y_t)
        else:
            yVals = jo.index_update(yVals, i, y_c + y_t)
    
    #making sure we have a split trailing edge
    if yVals[0] == yVals[-1]:
        val = yVals[0]
        yVals = jo.index_update(yVals, 0, val - 0.001)
        yVals = jo.index_update(yVals, len(yVals)-1, val + 0.001)
    return xVals, yVals
    
#yVals = fourDigitAirfoil(np.array([2, 4, 12], dtype=float))

def kulfan4(x):
    # x = [au1, au2, ... ,al1, al2, ...]
    order = 5
    au = np.array([0.0 for i in range(order)])
    al = np.array([0.0 for i in range(order)])
    au = jo.index_update(au, 0, x[0])
    au = jo.index_update(au, 1, x[1])
    au = jo.index_update(au, 2, x[2])
    au = jo.index_update(au, 3, x[3])
    au = jo.index_update(au, 4, x[4])
    al = jo.index_update(al, 0, x[5])
    al = jo.index_update(al, 1, x[6])
    al = jo.index_update(al, 2, x[7])
    al = jo.index_update(al, 3, x[8])
    al = jo.index_update(al, 4, x[9])
    
    n1 = 0.5
    n2 = 1.0
    teThickness = 0.001
    
    yVals = np.array([0.0 for x in range(len(xVals))])
    
    for i in range(len(yVals)):
        xC = xVals[i]
            
        if i > len(yVals)/2:
            su = 0
            for j in range(order):
                k_j = factorial(order)/(factorial(j)*factorial(order-j))
                s_j = k_j*(xC**j)*(1-xC)**(order-j)
                su = su + au[j]*s_j
            
            C = (xC**n1)*(1-xC)**n2
            yC = C*su + xC*teThickness
            
            yVals = jo.index_update(yVals, i, yC)
        else:
            sl = 0
            for j in range(order):
                k_j = factorial(order)/(factorial(j)*factorial(order-j))
                s_j = k_j*(xC**j)*(1-xC)**(order-j)
                sl = sl + al[j]*s_j
            
            C = (xC**n1)*(1-xC)**n2
            yC = C*sl + xC*teThickness
            yVals = jo.index_update(yVals, i, -yC)
    
    return xVals, yVals

# Bezier-PARSEC: An optimized aerofoil parameterization for design
def bezierParsec(x):
    xVals = np.array([0.0 for x in range(2*len(uVals) + 2*len(uValsBack))])
    yVals = np.array([0.0 for x in range(2*len(uVals) + 2*len(uValsBack))])
    uuVals = np.array([0.0 for x in range(2*len(uVals) + 2*len(uValsBack))])
    #x = [rle, b8, xt, yt, b15, dzte, betate, b0, b2, xc, yc, gammale, b17, zte, alphate]
    rle     = x[0]
    b8      = x[1]
    xt      = x[2]
    yt      = x[3]
    b15     = x[4]
    dzte    = x[5]
    betate  = x[6]
    b0      = x[7]
    b2      = x[8]
    xc      = x[9]
    yc      = x[10]
    gammale = x[11]
    b17     = x[12]
    zte     = x[13]
    alphate = x[14]
    
    for i in range(2*len(uVals) + 2*len(uValsBack)):
        
        if i < len(uValsBack) or i > ((2*len(uVals) + len(uValsBack)) - 1): #trailing edge
    
            if i < len(uValsBack):
                u = 1 - uValsBack[i]
            else:
                u = uValsBack[(i -(2*len(uVals) + len(uValsBack)))] + (1.0 - uValsBack[-1])
                
            x0 = xt
            x1 = (7.0*xt - 9.0*b8**2/(2.0*rle))/4.0
            #x1 = (b15-xt)*.33 + xt
            x2 = 3.0*xt - 15.0*b8**2.0/(4*rle)
            #x2 = (b15-xt)*.66 + xt
            x3 = b15
            x4 = 1
            y0 = yt
            y1 = yt 
            y2 = (yt+b8)/2
            y3 = dzte + (1-b15)*np.tan(betate)
            y4 = dzte
            
            xCoord = x0*(1-u)**4 + 4*x1*u*(1-u)**3 + 6*x2*(u**2)*(1-u)**2 +  4*x3*(u**3)*(1-u) + x4*u**4
            y_t = y0*(1-u)**4 + 4*y1*u*(1-u)**3 + 6*y2*(u**2)*(1-u)**2 +  4*y3*(u**3)*(1-u) + y4*u**4
                  
        else: #leading edge 
            if i < (len(uVals) + len(uValsBack)):
                u = uVals[-(i - len(uValsBack)) - 1]
            else:
                u = uVals[i - (len(uVals) + len(uValsBack))] + (1.0 - uVals[-1])
                
            if yt < (2.0*rle*xt/3.0)**0.5:
                b8bound = yt
            else:
                b8bound = (2.0*rle*xt/3.0)**0.5
            
            b8h = b8
            if b8h > b8bound:
                b8h = b8bound
            if b8h < 0.0:
                b8h = 0.0
                    
            x0 = 0
            x1 = 0
            x2 = 3.0*(b8h**2.0)/(2.0*rle)
            x3 = xt
            y0 = 0
            y1 = b8h
            y2 = yt
            y3 = yt
            
            xCoord = x0*(1-u)**3 + 3*x1*u*(1-u)**2 + 3*x2*(u**2)*(1-u) + x3*u**3        
            y_t = y0*(1-u)**3 + 3*y1*u*(1-u)**2 + 3*y2*(u**2)*(1-u) + y3*u**3
        
        if xCoord < xc:                    
            x0 = 0
            x1 = b0
            x2 = b2
            x3 = xc
            y0 = 0
            y1 = b0*np.tan(gammale)
            y2 = yc
            y3 = yc
            
            #iterative search for u 
            u = (xCoord)/(xc) 
            for j in range(3):
                f_x = x0*(1-u)**3 + 3*x1*u*(1-u)**2 + 3*x2*(u**2)*(1-u) + x3*u**3 - xCoord
                fp_x = -3*x0*(1-u)**2 + 3*x1*(1-u)**2 - 6*x1*u*(1-u) - 3*x2*u**2 + 6*x2*u*(1-u) + 3*x3*u**2
                u = u - f_x/fp_x
                                
            y_c = y0*(1-u)**3 + 3*y1*u*(1-u)**2 + 3*y2*(u**2)*(1-u) + y3*u**3
            
        else:
            x0 = xc
            x1 = (3.0*xc- yc/np.tan(gammale))/2.0
            x2 = (-8.0*yc/np.tan(gammale)+13.0*xc)/6.0
            # x1 = .7 #used for debugging
            # x2 = .9
            x3 = b17
            x4 = 1.0
            y0 = yc
            y1 = yc
            y2 = 5.0*yc/6.0
            y3 = zte - (1-b17)*np.tan(alphate)
            y4 = zte
            
            #iterative search for u
            u = (xCoord - xc)/(1.0 - xc)  
            for j in range(3):
                f_x = x0*(1-u)**4 + 4*x1*u*(1-u)**3 + 6*x2*(u**2)*(1-u)**2 +  4*x3*(u**3)*(1-u) + x4*u**4 - xCoord
                fp_x = -4*x0*(1-u)**3 + 4*x1*(1-u)**3 - 12*x1*u*(1-u)**2 + 12*x2*u*(1-u)**2 - 12*x2*(1-u)*u**2 + 12*x3*(1-u)*u**2 - 4*x3*u**3 + 4*x4*u**3
                
                if u < 0.25:
                    u = u - 1.0*f_x/fp_x
                else:
                    u = u - 1.0*f_x/fp_x
                    
            
            y_c = y0*(1-u)**4 + 4*y1*u*(1-u)**3 + 6*y2*(u**2)*(1-u)**2 +  4*y3*(u**3)*(1-u) + y4*u**4
        
        
        if i < (len(uVals) + len(uValsBack)):
            yVals = jo.index_update(yVals, i, y_c - y_t)
        else:
            yVals = jo.index_update(yVals, i, y_c + y_t)
        
        xVals = jo.index_update(xVals, i, xCoord)
        
        # uuVals = jo.index_update(uuVals, i, u) #used for debugging
        
    return xVals, yVals
    

            
def generateAirfoil(parameterizationType, parameters, outputFile):
    if parameterizationType == "NacaFourDigit":
        yValFn = fourDigitAirfoil
    if parameterizationType == "BezierParsec":
        yValFn = bezierParsec
    if parameterizationType == "Kulfan4":
        yValFn = kulfan4
    
    inX = np.array(parameters, dtype=float)
    xVals, yVals = yValFn(inX)
    
    f = open(outputFile, "w")
    for i in range(len(yVals)):
        # f.write(str(xVals[i]) + " " + str(yVals[i]) + " " + str(uuVals[i]) + "\n")
        f.write(str(xVals[i]) + " " + str(yVals[i]) +  "\n")
    f.close()
    
    #Returns jacobian of x values and jacobian of y values
    return jacfwd(yValFn)(inX)



                                 #x = [rle, b8, xt, yt, b15, dzte, betate, b0, b2, xc, yc, gammale, b17, zte, alphate]   
# af = generateAirfoil("BezierParsec", np.array([.03, .02, .3, .06, .95, .001, .4, .05, .2, .4, .03, .5, .98, 0.00, .01] ), "out.txt")

# af = generateAirfoil("Kulfan4", np.array([.13,.15,.15,.1,.08,.1,.13,.12,.1,.08] ), "out.txt")
       
        


