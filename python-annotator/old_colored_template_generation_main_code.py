
        #TODO: Remove this test code for generating the template
        #body's color map
        templateFrameOrigins = [(1, 38), (1, 85), (2, 27), (2, 94), (3, 62),
                        (4, 12), (4, 62), (5, 24)]
        #templateFrameOrigins = [(1, 38), (1, 85)]
        rotationValues = [0, 45, 90, 135, 180, 225, 270, 315]
        '''
        templateClouds = []
        for fileNumber, frameNumber in templateFrameOrigins:
            print "loading ", str(fileNumber), ",", str(frameNumber)
            self.cloudManager.seekTo(fileNumber, frameNumber)
            templateClouds.append(self.cloudManager.getCloud())

        #TODO: Use clearer terminology (which is the "template"?)

        #Great, now for each view, generate the appropriate approximating
        #orthographic projection for the view angle
        rasterViews = []
        for degrees in rotationValues:
            radians = float(degrees) * (3.1415 / 180.0)
            c = math.cos(radians)
            s = math.sin(radians)
            xvec = np.array([c, 0, s], dtype=np.float32)
            yvec = np.array([0, 1.0, 0.0], dtype=np.float32)
            bodyView = StandardBodyView(str(degrees), xvec, yvec)
            rasterViews.append(bodyView.get_smoothed_index())

        #Now, with each raster view, construct a corresponding projected template
        #point cloud
        projectedTemplates = []
        for rasterView in rasterViews:
            projectedTemplates.append(Rasterizer.pointIndexToCloud(rasterView))

        cloudEstimates = []

        #Great, now we have the collection of template clouds.
        #Take all of these, and perform rigid registration with the body template
        #bodyCloud = StandardBody.pointCloud
        for i in range(len(templateClouds)):
            bodyCloud = projectedTemplates[i]
            cloudPoints = templateClouds[i].getPoints()
            rotation = float(-rotationValues[i] + 180) * (3.1415 / 180.0)
            #Transform the cloudPoints to reflect the rotation prior,
            #and move the centroid to the origin
            c = math.cos(rotation)
            s = math.sin(rotation)
            M = np.array([[c, 0, -s],
                          [0, 1, 0],
                          [s, 0, c]], dtype=np.float32)
            rotatedCloudPoints = np.matmul(M, np.transpose(np.asarray(cloudPoints)))
            rotatedCloudPoints = np.transpose(rotatedCloudPoints)

            rotatedCloudCentroid = np.average(rotatedCloudPoints, axis=0)
            centeredRotatedCloudPoints = rotatedCloudPoints - rotatedCloudCentroid

            transformedCloud = pcl.PointCloud(centeredRotatedCloudPoints)

            max_iter = 1000
            icp = transformedCloud.make_IterativeClosestPoint()
            converged, transf, estimate, fitness = icp.icp(transformedCloud, bodyCloud, max_iter=max_iter)

            fig = plt.figure()
            #fig = matplotlib.figure.Figure()
            ax = fig.add_subplot(111, projection='3d')
            print converged, transf, estimate, fitness
            self.visualize(0, fitness, np.asarray(bodyCloud), np.asarray(estimate), ax)

            cloudEstimates.append((templateClouds[i], estimate, bodyCloud))

            #cloudPoints = np.asarray(templateCloud.getPoints())
            #DeformableReg.register(bodyPoints, cloudPoints, np.zeros((0,3)), np.zeros((0,3)))
        #self.cloudManager.seekTo(4, 62)
        '''

        #Save out the best rigid transforms to file
        cloudEstimatesFile = "CloudEstimates.pickle"
        '''if (os.path.isfile("CloudEstimates.pickle")):
            pass
        else:
            pickle.dump(cloudEstimates, open(cloudEstimatesFile, "wb"))
        '''
        #TODO: Continued experimentation spaghet, since the above stuff
        #took a while to generate
        cloudEstimates = pickle.load(open(cloudEstimatesFile, "rb"))
        #List of RGBPointCloud, PointCloud, PointCloud
        #where the first is the frame cloud, the second is the rigid-
        #transformed estimate, and the third is the standard body
        #projection to the estimated orthographic perspective
        print cloudEstimates

        #TODO: Remove me! Code for generating
        #deformable registrations between standard body
        #projections and captured images!
        '''
        registered = []

        count = 1

        for viewCloud, viewTransformed, templateCloud in cloudEstimates:
            origPoints = np.asarray(viewCloud.getPoints())
            transPoints = np.asarray(viewTransformed)
            templatePoints = np.asarray(templateCloud)
            if (count == 3 or count == 7):
                registered.append(None)
            else:
                reg = DeformableReg.register(origPoints, templatePoints, transPoints, np.zeros((0,3)), np.zeros((0,3)))
                registered.append(reg)
            count += 1

        registeredEstimatesFile = "RegisteredEstimates4.pickle"
        pickle.dump(registered, open(registeredEstimatesFile, "wb"))
        '''

        '''

        #Specify a list of lists of registered deformable point clouds
        #where each element can alternatively be "None"
        regFiles = ["RegisteredEstimates.pickle", "RegisteredEstimates2.pickle",
                    "RegisteredEstimates3.pickle", "RegisteredEstimates4.pickle"]
        #List is ordered by file
        regList = []
        for regFile in regFiles:
            regs = pickle.load(open(regFile, "rb"))
            regList.append(regs)

        #Great. Now, for each list of registrations,
        #loop through every point, and find the closest
        #point on the standard body. Associate to that index
        #the (distance, rgb color tuple) pair, and keep going
        #will aggregate the associations once this is done

        std_kd_tree = sp.spatial.cKDTree(StandardBody.pointArray)
        num_std_points = StandardBody.pointArray.shape[0]
        agg = []
        for _ in range(num_std_points):
            agg.append([])
        for regCollection in regList:
            for i in range(len(regCollection)):
                reg = regCollection[i]
                if reg is None:
                    continue
                num_reg_points = reg.shape[0]
                #Must not be none! Get the associated
                #frame data tuple from what we loaded in CloudEstimates
                RGBFrame, rigidEstimate, projection = cloudEstimates[i]
                #Okay, great. Now, get the frame's ordered color list
                frameColors = RGBFrame.getColors()
                #Go through every point in reg, and add distances to
                #agg based on the closest point in the standard body
                for j in range(num_reg_points):
                    K = 50
                    reg_point = reg[j]
                    color = frameColors[j]
                    (dists, inds) = std_kd_tree.query(reg_point, k=K)
                    for k in range(K):
                        d = dists[k]
                        ind = inds[k] 
                        agg[ind].append((d, color))

        nonColoredPoints = []
        nonColoredIndices = []

        coloredPoints = []
        coloredColors = []

        #Now that we've aggregated things, go through agg
        #and decide upon a color for each point
        colors = []
        for i in range(num_std_points):
            aggList = agg[i]
            totalWeight = 0
            sum_color = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for dist, color in aggList:
                thirdLife = 5.0
                weight = math.exp(-(dist / thirdLife))
                r, g, b, a = color
                colorArray = np.array([r, g, b, a], dtype=np.float32)
                sum_color += colorArray * weight
                totalWeight += weight
            if totalWeight == 0:
                #No color! assign a color of white for now,
                #and add to the not colored points
                colors.append((255, 255, 255, 255))
                nonColoredPoints.append(StandardBody.pointCloud[i])
                nonColoredIndices.append(i)
            else:
                color = sum_color / totalWeight
                colors.append(color)
                coloredPoints.append(StandardBody.pointCloud[i])
                coloredColors.append(color)

        #Great. Now, using the colored-in points,
        #generate a Kd tree
        coloredPoints = np.array(coloredPoints, dtype=np.float32)
        coloredTree = sp.spatial.cKDTree(coloredPoints)

        #Now, for every uncolored point, set its color to the
        #average of its k nearest neighbors
        K = 5
        for i in range(len(nonColoredPoints)):
            nonColoredPoint = nonColoredPoints[i]
            nonColoredIndex = nonColoredIndices[i]
            (dists, inds) = coloredTree.query(nonColoredPoint, k=K)
            totalWeight = 0
            sum_color = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for j in range(K):
                dist = dists[j]
                ind = inds[j]
                color = coloredColors[ind]
                thirdLife = 5.0
                weight = math.exp(-(dist / thirdLife))
                r, g, b, a = color
                colorArray = np.array([r, g, b, a], dtype=np.float32)
                sum_color += colorArray * weight
                totalWeight += weight
            color = sum_color / totalWeight
            colors[nonColoredIndex] = color
        

        #Cloud manager hax to show the standard body
        coloredBody = RGBPointCloud(0, StandardBody.pointCloud,
                                    colors, [])
        '''


        #pickle.dump(coloredBody, open(coloredTemplateFile, "wb"))
