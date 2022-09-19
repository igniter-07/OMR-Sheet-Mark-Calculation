import cv2
import utils
import numpy as np

################################

imgWidth = 700
imgHeight = 700
questions = 5
choices = 5
ans = [1, 2, 0, 0, 1]
webCamera = True
cameraNo = 1

################################

cap = cv2.VideoCapture(0)
cap.set(10, 150)

while cap.isOpened():
    if webCamera:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    img = cv2.resize(img, (imgWidth, imgHeight))  # RESIZE IMAGE

    # PREPROCESSING
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgContours = img.copy()
    imgFinal = img.copy()
    imgBlank = np.zeros((imgHeight, imgWidth, 3), np.uint8)  # CREATE A BLANK IMAGE FOR TESTING DEBUGGING IF REQUIRED
    imgBiggestContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 20)

    try:
        # FIND ALL CONTOURS AND DRAW THEM
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        # FIND CONTOURS WHICH ACTUALLY ARE RECTANGLES
        rectContourList = utils.rectangleContour(contours)
        biggestContour = utils.getCornerPoints(rectContourList[0])
        gradePoints = utils.getCornerPoints(rectContourList[1])

        if biggestContour.size!=0 and gradePoints.size!=0:
            cv2.drawContours(imgBiggestContours, biggestContour, -1, (0, 255, 0), 10)
            cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 10)

            # reordering points
            biggestContour = utils.reorder(biggestContour)
            gradePoints = utils.reorder(gradePoints)

            # getting the points
            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
            # generating transformation matrix
            transformationMatrix = cv2.getPerspectiveTransform(pt1, pt2)
            # Warp prespective
            imgWarpPres = cv2.warpPerspective(img, transformationMatrix, (imgWidth, imgHeight))

            # getting the points
            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            # generating transformation matrix
            transformationMatrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            # Warp prespective
            imgWarpPresG = cv2.warpPerspective(img, transformationMatrixG, (325, 150))

            # apply threshold
            imgWarpPresGray = cv2.cvtColor(imgWarpPres, cv2.COLOR_BGR2GRAY)
            imgThreshold = cv2.threshold(imgWarpPresGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = utils.splitBoxes(imgThreshold)  # boxes contains all bubbles
            myPixelValue = np.zeros((questions, choices))
            countCol = 0
            countRow = 0
            for images in boxes:
                totalPixel = cv2.countNonZero(images)
                myPixelValue[countRow][countCol] = totalPixel
                countCol = countCol + 1
                if countCol == choices:
                    countCol = 0
                    countRow = countRow + 1

            # Now traverse on myPixel arr to find the largest value pixel
            myIndex = []
            for x in range(0, questions):
                arr = myPixelValue[x]
                myIndexValue = np.where(arr == np.amax(arr))
                myIndex.append(myIndexValue[0][0])
            #print(myIndex)

            # Give grading
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                   grading.append(1)
                else:
                   grading.append(0)

            # final score
            score = (sum(grading)/questions) * 100

            # Display answers
            imgResult = imgWarpPres.copy()
            imgRaw = np.zeros_like(imgWarpPres)
            imgResult = utils.showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imgRaw = utils.showAnswers(imgRaw, myIndex, grading, ans, questions, choices)
            inverseTransformationMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInverseWarp = cv2.warpPerspective(imgRaw, inverseTransformationMatrix, (imgWidth, imgHeight))

            imgRawGrade = np.zeros_like(imgWarpPresG)
            cv2.putText(imgRawGrade, str(int(score))+"%", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), 3)
            inverseTransformationMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            # Warp prespective
            imgInverseRawGrade = cv2.warpPerspective(imgRawGrade, inverseTransformationMatrixG, (imgWidth, imgHeight))

            imgFinal = cv2.addWeighted(imgFinal, 1, imgInverseWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInverseRawGrade, 1, 0)
            # IMAGE ARRAY FOR DISPLAY
            imageArray = ([img, imgGray, imgCanny, imgContours],
                          [imgBigContour, imgThresh, imgWarpColored, imgFinal])
            cv2.imshow("Final Result", imgFinal)

    except:
        imageArray = ([img, imgGray, imgCanny, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # LABELS FOR DISPLAY
    lables = [["Original", "Gray", "Edges", "Contours"],
                  ["Biggest Contour", "Threshold", "Warpped", "Final"]]

    stackedImage = utils.stackImages(imageArray, 0.5, lables)
    cv2.imshow("Result", stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    k = cv2.waitKey(1)
    if k == ord("s") & 0xFF:
        cv2.imwrite("result" + str(count) + ".jpg", imgFinal)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
    if k == ord("q") & 0xFF:
        break

cap.release()
cv2.destroyAllWindows()